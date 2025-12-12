import streamlit as st
import requests
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import date

# ================================================================
# CONFIGURACIÓN GENERAL
# ================================================================

BASE_URL = "https://edgewspmfantasy.onrender.com"

# NFL: líneas y mercados por defecto
NFL_DEFAULT_LINES = {
    "passing_yards": 220.5,
    "rushing_yards": 55.5,
    "receiving_yards": 50.5,
}

NFL_KEY_MARKETS = {
    "QB": {"label": "Passing Yards", "market_type": "passing_yards"},
    "RB": {"label": "Rushing Yards", "market_type": "rushing_yards"},
    "WR": {"label": "Receiving Yards", "market_type": "receiving_yards"},
}

# NBA: de inicio solo “points” para TODOS los puestos
NBA_DEFAULT_LINES = {
    "points": 20.5,
}

NBA_KEY_MARKETS = {
    "PG": {"label": "Puntos", "market_type": "points"},
    "SG": {"label": "Puntos", "market_type": "points"},
    "SF": {"label": "Puntos", "market_type": "points"},
    "PF": {"label": "Puntos", "market_type": "points"},
    "C":  {"label": "Puntos", "market_type": "points"},
}

# Config centralizada por deporte
SPORT_CONFIG = {
    "NFL": {
        "api_prefix": "nfl",
        "emoji": "🏈",
        "label": "NFL",
        "default_season": 2024,
        "default_week": 16,
        "default_season_type": 2,
        "default_lines": NFL_DEFAULT_LINES,
        "key_markets": NFL_KEY_MARKETS,
    },
    "NBA": {
        "api_prefix": "nba",
        "emoji": "🏀",
        "label": "NBA",
        "default_season": 2024,
        # IMPORTANTE: para evitar el 422 del backend, usamos week>=1 y season_type>=1
        "default_week": 1,
        "default_season_type": 2,
        "default_lines": NBA_DEFAULT_LINES,
        "key_markets": NBA_KEY_MARKETS,
    },
}

# Tiers por edge (en unidad del mercado: yardas / puntos)
TIER_PLATINUM_EDGE = 25.0
TIER_PREMIUM_EDGE = 15.0
TIER_VALUE_EDGE = 8.0


# ================================================================
# FUNCIONES AUXILIARES WSPM – DIRECCIÓN / MARGEN / PROBABILIDAD
# ================================================================

def compute_direction_and_margin(
    wspm_projection: float,
    book_line: float,
) -> Tuple[str, float, float]:
    """
    Determina OVER/UNDER, margen absoluto y margen %.
    """
    if book_line <= 0:
        return "OVER", 0.0, 0.0

    if wspm_projection >= book_line:
        direction = "OVER"
        margin_yards = wspm_projection - book_line
    else:
        direction = "UNDER"
        margin_yards = book_line - wspm_projection

    margin_pct = (margin_yards / book_line) * 100.0
    return direction, margin_yards, margin_pct


def classify_confidence_from_margin(margin_pct: float) -> str:
    """
    Confianza a partir del margen %.
    Regla: Alta solo si margen > 15%.
    """
    if margin_pct > 15.0:
        return "Alta"
    elif margin_pct > 10.0:
        return "Media-Alta"
    elif margin_pct > 5.0:
        return "Media"
    else:
        return "Baja"


def estimate_prob_cover(margin_pct: float) -> float:
    """
    Heurística suave para P(cubrir) en función del margen %.
    Acotada entre 50% y 80%.
    """
    p = 0.5 + (margin_pct / 100.0) * 0.75
    if p < 0.5:
        p = 0.5
    if p > 0.8:
        p = 0.8
    return p


def build_wspm_markdown_explanation(
    sport: str,
    matchup: str,
    team: str,
    opp: str,
    player_name: str,
    position: str,
    market_type: str,
    wspm_projection: float,
    book_line: float,
    safety_pct_backend: float,
) -> Tuple[str, str, float, float, str, float]:
    """
    Construye el bloque Markdown explicativo WSPM.
    Devuelve:
      markdown, direction, margin_yards, effective_margin_pct, confidence_label, prob_cover_pct
    """

    direction, margin_yards, margin_pct = compute_direction_and_margin(
        wspm_projection, book_line
    )

    effective_margin_pct = (
        safety_pct_backend
        if safety_pct_backend and safety_pct_backend > 0
        else margin_pct
    )

    confidence = classify_confidence_from_margin(effective_margin_pct)
    prob_cover = estimate_prob_cover(effective_margin_pct) * 100.0
    net_adjust = wspm_projection - book_line

    # Distribución narrativa de ese ajuste
    if net_adjust == 0:
        matchup_adj = volume_adj = risk_adj = tempo_adj = 0.0
    else:
        matchup_adj = 0.40 * net_adjust
        volume_adj = 0.35 * net_adjust
        risk_adj = -0.20 * abs(net_adjust)
        tempo_adj = net_adjust - (matchup_adj + volume_adj + risk_adj)

    market_label_map = {
        "passing_yards": "yardas por pase",
        "rushing_yards": "yardas por tierra",
        "receiving_yards": "yardas por recepción",
        "points": "puntos",
        "rebounds": "rebotes",
        "assists": "asistencias",
    }
    market_label = market_label_map.get(
        market_type,
        market_type.replace("_", " ")
    )

    markdown = f"""### 📈 Proyección del Modelo WSPM ({sport})

*Partido:* {team} vs {opp} — {matchup}  
*Jugador:* **{player_name}** – *Posición:* {position}  
*Línea del book (Proyección O/U):* **{book_line:.1f} {market_label}**  
*Proyección del modelo WSPM:* **{wspm_projection:.1f} {market_label}**  
*Pick del modelo WSPM:* **{direction} {book_line:.1f} {market_label}**

#### ⚖️ Ponderación de Variables Clave (vs Línea Base)

* **Variable 1: Matchup Defensivo Avanzado (DVOA/YAC / On-Off / Defensive Rating):**
  - *Ponderación estimada:* {matchup_adj:+.1f} {market_label}
* **Variable 2: Volumen de Juego Proyectado (Targets/Carries/Pases / Usage):**
  - *Ponderación estimada:* {volume_adj:+.1f} {market_label}
* **Variable 3: Riesgo / Reversión de Margen:** volatilidad, blowout, cambios de script, faltas, etc.
  - *Ponderación estimada:* {risk_adj:+.1f} {market_label}
* **Variable 4: Game Flow (Ritmo / Tempo del Partido):** total del partido y ritmo ofensivo.
  - *Ponderación estimada:* {tempo_adj:+.1f} {market_label}

#### 🎯 Análisis y Justificación

*Ajuste Neto Total (suma de ponderaciones):* **{net_adjust:+.1f} {market_label}**  
*Margen de Seguridad (WSPM):* **{margin_yards:.1f} {market_label}** (~{effective_margin_pct:.1f}% sobre la línea)  
*Probabilidad Estimada de Cubrir la Línea:* **{prob_cover:.1f}%**

#### 💡 Conclusión

*Dirección Esperada (Valor WSPM):* **{direction} {book_line:.1f} {market_label}**  
*Confianza del Pick (Rigurosa):* **{confidence}**  
"""

    return markdown, direction, margin_yards, effective_margin_pct, confidence, prob_cover


# ================================================================
# LLAMADAS A LA API (CACHEADAS)
# ================================================================

def _get_api_base(sport: str) -> str:
    prefix = SPORT_CONFIG[sport]["api_prefix"]
    return f"{BASE_URL}/api/v1/{prefix}"


@st.cache_data(ttl=3600)
def get_games_with_odds(
    sport: str,
    season: int,
    season_type: int,
    week: int,
    date_str: str,
) -> Optional[List[Dict[str, Any]]]:
    """
    NFL: usa season / season_type / week
    NBA: usa date_str (YYYYMMDD)
    """
    try:
        api_base = _get_api_base(sport)
        url = f"{api_base}/games-with-odds"

        if sport == "NFL":
            params = {
                "season": season,
                "season_type": season_type,
                "week": week,
            }
        else:  # NBA
            params = {"date": date_str}

        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("games", [])
    except Exception as e:
        st.error(f"Error al obtener juegos ({sport}): {e}")
        return None


@st.cache_data(ttl=3600)
def get_team_roster(sport: str, team_abbr: str) -> Optional[Dict[str, Any]]:
    """Roster NFL/NBA según deporte."""
    try:
        api_base = _get_api_base(sport)
        url = f"{api_base}/team/{team_abbr}/roster"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error al obtener roster de {team_abbr} ({sport}): {e}")
        return None


def call_player_projection(sport: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Llama al auto-projection-report del deporte correspondiente."""
    api_base = _get_api_base(sport)
    url = f"{api_base}/wspm/auto-projection-report"

    try:
        r = requests.post(url, json=payload, timeout=60)

        if r.status_code == 422:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            st.warning(f"[{sport}] 422 Unprocessable Entity: {detail}")
            with st.expander("Payload enviado (debug)"):
                st.json(payload)
            return None

        if r.status_code >= 400:
            try:
                error_detail = r.json().get("detail", r.text)
            except Exception:
                error_detail = r.text
            st.error(f"[{sport}] Error {r.status_code} en la API: {error_detail}")
            with st.expander("Payload enviado (debug)"):
                st.json(payload)
            return None

        return r.json()

    except requests.exceptions.RequestException as e:
        st.error(f"[{sport}] Error de conexión con la API: {e}")
        return None
    except Exception as e:
        st.error(f"[{sport}] Error inesperado: {e}")
        return None


# ================================================================
# UTILIDADES DE DATOS
# ================================================================

def get_player_data(roster: Dict[str, Any], athlete_id: str) -> Optional[Dict[str, Any]]:
    if not roster or "players" not in roster:
        return None
    for p in roster["players"]:
        if str(p.get("athlete_id")) == str(athlete_id):
            return p
    return None


def find_player_market_type(sport: str, position: str) -> str:
    pos = (position or "").upper()

    if sport == "NFL":
        if pos == "QB":
            return "passing_yards"
        if pos == "RB":
            return "rushing_yards"
        if pos == "WR":
            return "receiving_yards"
        return "unknown"

    # NBA – por ahora todo "points"
    if sport == "NBA":
        return "points"

    return "unknown"


def build_player_payload(
    sport: str,
    event_id: str,
    player: Dict[str, Any],
    team_abbr: str,
    opp_team_abbr: str,
    market_type: str,
    book_line: float,
    season: int,
    season_type: int,
    week: int,
) -> Dict[str, Any]:
    """
    Payload común para NFL/NBA.
    Para NBA forzamos season_type>=1 y week>=1 para evitar 422.
    """
    if sport == "NBA":
        if season_type < 1:
            season_type = 1
        if week < 1:
            week = 1

    return {
        "sport": sport.lower(),  # opcional, si lo usas en backend
        "athlete_id": str(player["athlete_id"]),
        "event_id": str(event_id),
        "season": int(season),
        "season_type": int(season_type),
        "week": int(week),
        "player_name": player["name"],
        "player_team": team_abbr,
        "opponent_team": opp_team_abbr,
        "position": player.get("position"),
        "market_type": market_type,
        "book_line": float(book_line),
    }


def summarize_projection(prediction: Dict[str, Any]) -> Tuple[float, float, float, float, str]:
    """
    Extrae métricas clave del JSON de WSPM.
    Devuelve: (wspm_projection, book_line, edge, safety_margin_pct, tier)
    """
    wspm_proj = prediction.get("wspm_projection")
    if wspm_proj is None:
        wspm_proj = prediction.get("model_projection") or prediction.get("projection")
    wspm_proj = float(wspm_proj or 0.0)

    book_line = prediction.get("book_line")
    if book_line is None:
        book_line = prediction.get("input_book_line")
    book_line = float(book_line or 0.0)

    edge = prediction.get("edge")
    if edge is None and book_line:
        edge = wspm_proj - book_line
    edge = float(edge or 0.0)

    safety_pct = float(prediction.get("safety_margin_pct") or 0.0)

    if edge >= TIER_PLATINUM_EDGE:
        tier = "Platinum"
    elif edge >= TIER_PREMIUM_EDGE:
        tier = "Premium"
    elif edge >= TIER_VALUE_EDGE:
        tier = "Value"
    else:
        tier = "Leans"

    return wspm_proj, book_line, edge, safety_pct, tier


def group_roster_key_players(roster: Dict[str, Any], sport: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Heurística simple para elegir jugadores clave.
    NFL: 1 QB, 2 RB, 2 WR
    NBA: primeros 3 jugadores del roster
    """
    players = roster.get("players", []) if roster else []
    result: Dict[str, List[Dict[str, Any]]] = {}

    if sport == "NFL":
        grouped = {"QB": [], "RB": [], "WR": []}
        for p in players:
            pos = (p.get("position") or "").upper()
            if pos in grouped:
                grouped[pos].append(p)

        if grouped["QB"]:
            result["QB"] = grouped["QB"][:1]
        if grouped["RB"]:
            result["RB"] = grouped["RB"][:2]
        if grouped["WR"]:
            result["WR"] = grouped["WR"][:2]

    else:  # NBA u otros – tomamos 3 primeros
        result["ALL"] = players[:3]

    return result


def tier_emoji(tier: str) -> str:
    t = tier.lower()
    if t == "platinum":
        return "💎"
    if t == "premium":
        return "🎯"
    if t == "value":
        return "📈"
    return "➖"


# ================================================================
# AUTO PICKS SEMANALES – SOLO NFL
# ================================================================

def auto_scan_game(
    sport: str,
    game: Dict[str, Any],
    season: int,
    season_type: int,
    week: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Escanea un partido NFL:
      - carga roster home / away
      - elige jugadores clave
      - llama WSPM auto-projection-report
    Devuelve lista de picks + lista de errores.
    """
    event_id = game["event_id"]
    home = game["home_team"]
    away = game["away_team"]
    home_abbr = home["abbr"]
    away_abbr = away["abbr"]

    default_lines = SPORT_CONFIG[sport]["default_lines"]

    picks: List[Dict[str, Any]] = []
    errors: List[str] = []

    def process_team(team_abbr: str, opp_abbr: str):
        nonlocal picks, errors
        roster = get_team_roster(sport, team_abbr)
        if not roster:
            errors.append(f"Error: no se pudo cargar roster de {team_abbr}.")
            return

        key_players = group_roster_key_players(roster, sport)

        for _, plist in key_players.items():
            for player in plist:
                pos = player.get("position") or ""
                mkt_type = find_player_market_type(sport, pos)
                if mkt_type == "unknown":
                    continue

                default_line = default_lines.get(mkt_type, 50.5)

                payload = build_player_payload(
                    sport=sport,
                    event_id=event_id,
                    player=player,
                    team_abbr=team_abbr,
                    opp_team_abbr=opp_abbr,
                    market_type=mkt_type,
                    book_line=default_line,
                    season=season,
                    season_type=season_type,
                    week=week,
                )

                proj = call_player_projection(sport, payload)
                if not proj:
                    errors.append(
                        f"[INFO] Ignorando {player['name']} ({team_abbr}, {pos}) por error en proyección."
                    )
                    continue

                wspm_proj, book_line, edge, safety_pct, tier = summarize_projection(proj)

                direction, margin_yards, margin_pct = compute_direction_and_margin(
                    wspm_proj, book_line
                )
                prob_cover = estimate_prob_cover(margin_pct) * 100.0
                confidence = classify_confidence_from_margin(margin_pct)

                picks.append(
                    {
                        "matchup": game["matchup"],
                        "provider": game["odds"].get("provider"),
                        "details": game["odds"].get("details"),
                        "over_under": game["odds"].get("over_under"),
                        "event_id": event_id,
                        "team": team_abbr,
                        "opp": opp_abbr,
                        "athlete_id": player["athlete_id"],
                        "name": player["name"],
                        "position": pos,
                        "market_type": mkt_type,
                        "book_line": book_line,
                        "wspm_projection": wspm_proj,
                        "edge": edge,
                        "safety_margin_pct": safety_pct,
                        "tier": tier,
                        "direction": direction,
                        "margin_yards": margin_yards,
                        "margin_pct": margin_pct,
                        "prob_cover": prob_cover,
                        "confidence": confidence,
                        "raw_projection": proj,
                    }
                )

    process_team(home_abbr, away_abbr)
    process_team(away_abbr, home_abbr)

    return picks, errors


def build_week_auto_report_text(
    week: int,
    season: int,
    season_type: int,
    max_games: Optional[int],
) -> Tuple[str, pd.DataFrame]:
    """Solo NFL: recorre games-with-odds, llama WSPM y arma reporte para redes."""
    sport = "NFL"
    games = get_games_with_odds(
        sport=sport,
        season=season,
        season_type=season_type,
        week=week,
        date_str="",  # no aplica NFL
    )
    if not games:
        return f"Sin partidos para week={week}, season={season}.", pd.DataFrame()

    if max_games and max_games > 0:
        games = games[:max_games]

    all_picks: List[Dict[str, Any]] = []
    all_errors: List[str] = []

    for g in games:
        picks, errs = auto_scan_game(sport, g, season, season_type, week)
        all_picks.extend(picks)
        all_errors.extend(errs)

    if not all_picks:
        txt = "No se generaron picks válidos (errores o 422 en todos los intentos).\n\n"
        if all_errors:
            txt += "Log de errores:\n" + "\n".join(all_errors)
        return txt, pd.DataFrame()

    df = pd.DataFrame(all_picks)
    df_sorted = df.sort_values(
        by=["tier", "prob_cover", "edge"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    top = df_sorted.iloc[0]
    free_text = (
        f"🎯 Free Pick WSPM – Semana {week} NFL {season}\n\n"
        f"{top['matchup']} | {top['provider']}: {top['details']}, O/U {top['over_under']}\n"
        f"{top['name']} ({top['team']} – {top['market_type'].replace('_', ' ')}) "
        f"{top['direction']} {top['book_line']:.1f} | Proyección WSPM: {top['wspm_projection']:.1f}, "
        f"edge {top['edge']:.1f}, {top['margin_pct']:.1f}% margen, "
        f"Confianza {top['confidence']} (P(cubrir) ~ {top['prob_cover']:.1f}%)\n"
    )

    lines_premium: List[str] = []
    for idx, row in df_sorted.iterrows():
        emoji = tier_emoji(row["tier"])
        line = (
            f"{idx+1}. {emoji} {row['name']} ({row['team']} – "
            f"{row['market_type'].replace('_', ' ')}) {row['direction']} {row['book_line']:.1f} | "
            f"proj {row['wspm_projection']:.1f}, edge {row['edge']:.1f}, "
            f"{row['margin_pct']:.1f}% margen, P(cubrir)~{row['prob_cover']:.1f}% – "
            f"Tier {row['tier']} / Conf {row['confidence']} "
            f"[{row['matchup']} | {row['provider']}: {row['details']}, O/U {row['over_under']}]"
        )
        lines_premium.append(line)

    premium_block = (
        f"\n💎 Premium / Platinum WSPM – Semana {week} NFL {season}\n"
        f"Top edges del modelo (ordenados por value esperado y probabilidad):\n\n"
        + "\n".join(lines_premium)
    )

    breakdown_lines: List[str] = []
    breakdown_lines.append(f"\n📊 Breakdown por partido – Semana {week} NFL {season}\n")

    for matchup in df_sorted["matchup"].unique():
        sub = df_sorted[df_sorted["matchup"] == matchup]
        if sub.empty:
            continue
        header = (
            f"{matchup} | {sub.iloc[0]['provider']}: {sub.iloc[0]['details']}, "
            f"O/U {sub.iloc[0]['over_under']}"
        )
        breakdown_lines.append(header)
        breakdown_lines.append("-" * len(header))
        for _, row in sub.iterrows():
            emoji = tier_emoji(row["tier"])
            breakdown_lines.append(
                f"  {emoji} {row['name']} ({row['team']} – {row['market_type'].replace('_', ' ')}) "
                f"{row['direction']} {row['book_line']:.1f} | proj {row['wspm_projection']:.1f}, "
                f"edge {row['edge']:.1f}, {row['margin_pct']:.1f}% margen, "
                f"P(cubrir)~{row['prob_cover']:.1f}%, Tier {row['tier']} ({row['confidence']})"
            )
        breakdown_lines.append("")

    full_text = free_text + premium_block + "\n" + "\n".join(breakdown_lines)

    if all_errors:
        full_text += "\n\nLog de errores WSPM/rosters:\n" + "\n".join(all_errors)

    return full_text, df_sorted


# ================================================================
# INTERFAZ STREAMLIT
# ================================================================

def main():
    st.set_page_config(
        page_title="WSPM Edge – NFL & NBA",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --------------------------
    # Sidebar – deporte y config
    # --------------------------
    st.sidebar.header("⚙️ Configuración Global")

    sport = st.sidebar.radio(
        "Deporte",
        options=["NFL", "NBA"],
        index=0,
        horizontal=True,
    )

    cfg = SPORT_CONFIG[sport]

    st.sidebar.markdown(f"**Backend WSPM:**")
    st.sidebar.code(BASE_URL, language="bash")

    if sport == "NFL":
        season = st.sidebar.number_input(
            "Temporada NFL",
            value=cfg["default_season"],
            min_value=2018,
            step=1,
        )
        week = st.sidebar.number_input(
            "Semana NFL",
            value=cfg["default_week"],
            min_value=1,
            max_value=22,
            step=1,
        )
        season_type = st.sidebar.selectbox(
            "Tipo de temporada",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Preseason (1)", 2: "Regular (2)", 3: "Postseason (3)"}[x],
            index=1,
        )
        date_str = ""  # no aplica
    else:
        season = st.sidebar.number_input(
            "Temporada NBA (año)",
            value=cfg["default_season"],
            min_value=2015,
            step=1,
        )
        game_date = st.sidebar.date_input(
            "Fecha de los partidos NBA",
            value=date.today(),
        )
        date_str = game_date.strftime("%Y%m%d")
        week = cfg["default_week"]              # 1
        season_type = cfg["default_season_type"]  # 2

    # --------------------------
    # Título principal
    # --------------------------
    st.title(f"{cfg['emoji']} WSPM Edge – {sport} (Player Props & Auto Picks)")
    st.markdown(
        "Frontend en Streamlit conectado a tu API WSPM para **proyecciones, edges, "
        "probabilidad de cubrir y textos listos para redes sociales**."
    )
    st.divider()

    # Obtener juegos para el deporte seleccionado
    games = get_games_with_odds(
        sport=sport,
        season=int(season),
        season_type=int(season_type),
        week=int(week),
        date_str=date_str,
    )
    if not games:
        st.error("No se pudieron cargar juegos desde /games-with-odds.")
        return

    game_options = {
        f"{g['matchup']} ({g['odds'].get('details', 'N/A')}, O/U {g['odds'].get('over_under', 'N/A')})": g
        for g in games
    }
    game_names = list(game_options.keys())

    tab1, tab2 = st.tabs(
        ["🎯 Player Prop (manual)", "🤖 Auto Picks Semana (Free/Premium/Platinum)"]
    )

    # ============================================================
    # TAB 1 – Player Prop manual (NFL + NBA)
    # ============================================================
    with tab1:
        st.header(f"🎯 Player Prop – Análisis Individual ({sport})")

        selected_game_name = st.selectbox(
            "Selecciona un partido:",
            game_names,
            index=0,
            key=f"game_select_tab1_{sport}",
        )

        selected_game = game_options[selected_game_name]
        event_id = selected_game["event_id"]
        home_abbr = selected_game["home_team"]["abbr"]
        away_abbr = selected_game["away_team"]["abbr"]
        matchup_text = selected_game["matchup"]

        st.info(
            f"Partido Seleccionado: **{matchup_text}** | Event ID: `{event_id}`"
        )

        col_team, col_player = st.columns(2)

        with col_team:
            team_choice = st.radio(
                "Equipo para analizar",
                options=[home_abbr, away_abbr],
                index=0,
                horizontal=True,
            )
            opp_choice = away_abbr if team_choice == home_abbr else home_abbr
            st.write(f"Oponente: **{opp_choice}**")

            roster_data = get_team_roster(sport, team_choice)

        players_for_selection: List[Dict[str, Any]] = []

        if not roster_data:
            st.warning(f"No se pudo cargar el roster para {team_choice}.")
        else:
            default_lines = cfg["default_lines"]

            for p in roster_data.get("players", []):
                pos = p.get("position")
                mkt_type = find_player_market_type(sport, pos)
                if mkt_type != "unknown":
                    players_for_selection.append(
                        {
                            "id": p["athlete_id"],
                            "name": f"{p['name']} ({pos})",
                            "position": pos,
                            "market_type": mkt_type,
                            "default_line": default_lines.get(mkt_type),
                        }
                    )

            with col_player:
                if not players_for_selection:
                    st.error("No se encontraron jugadores válidos en este roster.")
                else:
                    player_df = pd.DataFrame(players_for_selection)
                    player_options = player_df["name"].tolist()
                    selected_player_name = st.selectbox(
                        "Jugador a analizar",
                        player_options,
                        index=0,
                    )
                    selected_player_row = player_df[
                        player_df["name"] == selected_player_name
                    ].iloc[0]
                    selected_athlete_id = selected_player_row["id"]
                    selected_position = selected_player_row["position"]
                    selected_market_type = selected_player_row["market_type"]
                    selected_default_line = float(selected_player_row["default_line"])

        st.divider()

        if roster_data and players_for_selection:
            st.subheader("2. Línea del Book y Mercado")

            col_market, col_line = st.columns([1, 1])

            key_markets = cfg["key_markets"]
            market_conf = key_markets.get(
                (selected_position or "").upper(),
                {"label": selected_market_type},
            )

            with col_market:
                market_label = market_conf["label"]
                st.metric(
                    label="Mercado a Predecir",
                    value=market_label,
                    delta=selected_market_type.replace("_", " ").title(),
                )

            with col_line:
                book_line = st.number_input(
                    "Línea de Apuesta (Book Line)",
                    min_value=0.0,
                    value=selected_default_line,
                    step=0.5,
                    help=f"Línea O/U actual para {market_label}.",
                )

            st.divider()

            if st.button("🚀 Ejecutar Proyección WSPM", type="primary"):
                player_info = get_player_data(roster_data, selected_athlete_id)
                if not player_info:
                    st.error("Error: No se encontró la información completa del jugador.")
                else:
                    payload = build_player_payload(
                        sport=sport,
                        event_id=event_id,
                        player=player_info,
                        team_abbr=team_choice,
                        opp_team_abbr=opp_choice,
                        market_type=selected_market_type,
                        book_line=book_line,
                        season=int(season),
                        season_type=int(season_type),
                        week=int(week),
                    )

                    prediction_report = call_player_projection(sport, payload)

                    st.header("📊 Resultado de la Proyección")

                    if prediction_report:
                        wspm_proj, used_line, edge, safety_pct_backend, tier = (
                            summarize_projection(prediction_report)
                        )

                        wspm_md, direction, margin_yards, margin_pct, conf_label, prob_cover = (
                            build_wspm_markdown_explanation(
                                sport=sport,
                                matchup=matchup_text,
                                team=team_choice,
                                opp=opp_choice,
                                player_name=player_info["name"],
                                position=player_info.get("position", ""),
                                market_type=selected_market_type,
                                wspm_projection=wspm_proj,
                                book_line=used_line,
                                safety_pct_backend=safety_pct_backend,
                            )
                        )

                        col_proj, col_edge, col_safety, col_prob, col_conf = st.columns(5)

                        with col_proj:
                            st.metric("Proyección WSPM", f"{wspm_proj:.1f}")
                        with col_edge:
                            st.metric(
                                "Edge vs. Book",
                                f"{edge:+.1f}",
                                delta=f"{edge:+.1f}",
                            )
                        with col_safety:
                            st.metric("Margen % (modelo)", f"{margin_pct:.1f}%")
                        with col_prob:
                            st.metric("P(cubrir) estimada", f"{prob_cover:.1f}%")
                        with col_conf:
                            st.metric("Confianza Pick", conf_label)

                        st.markdown(wspm_md)

                        st.subheader("📣 Texto corto para redes (Player Prop)")
                        short_txt = (
                            f"{tier_emoji(tier)} WSPM – {matchup_text}\n"
                            f"{player_info['name']} ({team_choice} – {selected_market_type.replace('_',' ')}) "
                            f"{direction} {used_line:.1f}\n"
                            f"Proy: {wspm_proj:.1f} | Edge {edge:+.1f} | "
                            f"Margen {margin_pct:.1f}% | P(cubrir)~{prob_cover:.1f}% | Conf {conf_label}\n"
                            f"#{sport} #PlayerProps #WSPM"
                        )
                        st.code(short_txt, language="markdown")
                    else:
                        st.error("La API no devolvió una proyección válida.")

    # ============================================================
    # TAB 2 – Auto Picks Semana (solo NFL)
    # ============================================================
    with tab2:
        if sport != "NFL":
            st.info("Los Auto Picks semanales (Free/Premium/Platinum) están disponibles por ahora solo para NFL.")
        else:
            st.header("🤖 Auto Picks WSPM – Semana Completa (NFL)")

            col_conf, col_btn = st.columns([3, 1])

            with col_conf:
                max_games = st.number_input(
                    "Máximo de partidos a escanear (0 = todos)",
                    min_value=0,
                    max_value=len(games),
                    value=min(4, len(games)),
                    step=1,
                )

            with col_btn:
                run_week = st.button("Generar reporte semanal", type="primary")

            if run_week:
                with st.spinner("Generando picks de la semana con WSPM..."):
                    report_text, df_picks = build_week_auto_report_text(
                        week=int(week),
                        season=int(season),
                        season_type=int(season_type),
                        max_games=int(max_games),
                    )

                st.subheader("📣 Texto para redes (Free + Premium + Breakdown)")
                st.code(report_text, language="markdown")

                if not df_picks.empty:
                    st.subheader(
                        "📑 Tabla de Picks (ordenados por tier / probabilidad / edge)"
                    )
                    st.dataframe(
                        df_picks[
                            [
                                "matchup",
                                "team",
                                "name",
                                "position",
                                "market_type",
                                "direction",
                                "book_line",
                                "wspm_projection",
                                "edge",
                                "margin_pct",
                                "prob_cover",
                                "tier",
                                "confidence",
                            ]
                        ],
                        use_container_width=True,
                    )

                    st.markdown(
                        "_Tip_: filtra en la tabla y copia solo los Platinum/Premium con mayor "
                        "probabilidad para tu newsletter o canal de Telegram."
                    )


if __name__ == "__main__":
    main()
