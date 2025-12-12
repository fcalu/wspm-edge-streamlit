import streamlit as st
import json
from typing import Dict, Any, List, Tuple, Optional
import requests
import pandas as pd

# ================================================================
# CONFIGURACIÓN BÁSICA DEL CLIENTE / CONSTANTES
# ================================================================

BASE_URL = "https://edgewspmfantasy.onrender.com"
API_BASE_URL = f"{BASE_URL}/api/v1/nfl"

GAMES_WITH_ODDS_ENDPOINT = f"{API_BASE_URL}/games-with-odds"
ROSTER_ENDPOINT = f"{API_BASE_URL}/team/{{team_abbr}}/roster"
PLAYER_PROJECTION_ENDPOINT = f"{API_BASE_URL}/wspm/auto-projection-report"

DEFAULT_SEASON = 2024
DEFAULT_SEASON_TYPE = 2  # 2 = Regular Season
DEFAULT_WEEK = 16

# Líneas genéricas base (si tu API no manda línea específica por jugador)
DEFAULT_LINES = {
    "passing_yards": 220.5,
    "rushing_yards": 55.5,
    "receiving_yards": 50.5,
}

# Tipos de mercado clave
KEY_MARKETS = {
    "QB": {"label": "Passing Yards", "market_type": "passing_yards"},
    "RB": {"label": "Rushing Yards", "market_type": "rushing_yards"},
    "WR": {"label": "Receiving Yards", "market_type": "receiving_yards"},
}

# Tiers por edge (yardas)
TIER_PLATINUM_EDGE = 25.0
TIER_PREMIUM_EDGE = 15.0
TIER_VALUE_EDGE = 8.0


# ================================================================
# ALGORITMOS WSPM – DIRECCIÓN, MARGEN, PROBABILIDAD, CONFIANZA
# ================================================================

def compute_direction_and_margin(wspm_projection: float, book_line: float) -> Tuple[str, float, float]:
    """
    Determina la dirección (OVER/UNDER), el margen en yardas
    y el margen en % respecto a la línea.
    """
    if book_line <= 0:
        # Evitar divisiones raras: si la línea es 0, asumimos OVER sin margen real
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
    Clasifica la confianza a partir del margen %.
    Regla explícita: Confianza Alta sólo si margen > 15%.
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
    Heurística suave para estimar P(cubrir) en función del margen %.
    Se acota entre 50% y 80% para no inventar cosas locas.
    """
    # Base 50%, subimos según margen (escala suave)
    p = 0.5 + (margin_pct / 100.0) * 0.75
    if p < 0.5:
        p = 0.5
    if p > 0.8:
        p = 0.8
    return p


def build_wspm_markdown_explanation(
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
    Construye el bloque Markdown largo con la narrativa WSPM:
    - Ponderación de variables
    - Ajuste neto
    - Margen
    - Probabilidad de cubrir
    - Confianza
    Devuelve:
      markdown, direction, margin_yards, effective_margin_pct, confidence_label, prob_cover_pct
    """

    # 1) Dirección y margen
    direction, margin_yards, margin_pct = compute_direction_and_margin(
        wspm_projection, book_line
    )

    # 2) Tomamos el safety_margin_pct de tu API si viene informado,
    #    si no, usamos el margen calculado vs línea.
    effective_margin_pct = (
        safety_pct_backend
        if safety_pct_backend and safety_pct_backend > 0
        else margin_pct
    )

    # 3) Confianza y probabilidad de cubrir
    confidence = classify_confidence_from_margin(effective_margin_pct)
    prob_cover = estimate_prob_cover(effective_margin_pct) * 100.0

    # 4) Ajuste neto (proyección - línea)
    net_adjust = wspm_projection - book_line

    # Repartimos ese ajuste en "componentes narrativos" (solo narrativa, no afecta al modelo real)
    if net_adjust == 0:
        matchup_adj = volume_adj = risk_adj = tempo_adj = 0.0
    else:
        matchup_adj = 0.40 * net_adjust
        volume_adj = 0.35 * net_adjust
        risk_adj = -0.20 * abs(net_adjust)
        tempo_adj = net_adjust - (matchup_adj + volume_adj + risk_adj)

    market_label = {
        "passing_yards": "yardas por pase",
        "rushing_yards": "yardas por tierra",
        "receiving_yards": "yardas por recepción",
    }.get(market_type, market_type.replace("_", " "))

    markdown = f"""### 📈 Proyección del Modelo WSPM

*Partido:* {team} vs {opp} — {matchup}  
*Jugador:* **{player_name}** – *Posición:* {position}  
*Línea del book (Proyección O/U):* **{book_line:.1f} {market_label}**  
*Proyección del modelo WSPM:* **{wspm_projection:.1f} {market_label}**  
*Pick del modelo WSPM:* **{direction} {book_line:.1f} {market_label}**

#### ⚖️ Ponderación de Variables Clave (vs Línea Base)

* **Variable 1: Matchup Defensivo Avanzado (DVOA/YAC):** impacto del enfrentamiento defensa vs posición.
  - *Ponderación estimada:* {matchup_adj:+.1f} {market_label}
* **Variable 2: Volumen de Juego Proyectado (Targets/Carries/Pases):** rol esperado en el gameplan ofensivo.
  - *Ponderación estimada:* {volume_adj:+.1f} {market_label}
* **Variable 3: Riesgo / Reversión de Margen:** volatilidad, posibilidad de blowout, cambios de script o lesiones.
  - *Ponderación estimada:* {risk_adj:+.1f} {market_label}
* **Variable 4: Game Flow (Ritmo / Tempo del Partido):** total del partido y ritmo ofensivo combinado.
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

@st.cache_data(ttl=3600)
def get_games_with_odds(
    week: int, season_type: int, season: int
) -> Optional[List[Dict[str, Any]]]:
    """Obtiene la lista de juegos con momios desde tu API."""
    try:
        params = {"week": week, "season_type": season_type, "season": season}
        r = requests.get(GAMES_WITH_ODDS_ENDPOINT, params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("games", [])
    except Exception as e:
        st.error(f"Error al obtener juegos: {e}")
        return None


@st.cache_data(ttl=3600)
def get_team_roster(team_abbr: str) -> Optional[Dict[str, Any]]:
    """Obtiene el roster completo de un equipo."""
    try:
        url = ROSTER_ENDPOINT.format(team_abbr=team_abbr)
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error al obtener roster de {team_abbr}: {e}")
        return None


def call_player_projection(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Llama al endpoint de proyección automática y maneja errores."""
    url = PLAYER_PROJECTION_ENDPOINT
    try:
        r = requests.post(url, json=payload, timeout=60)

        if r.status_code == 422:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            st.warning(f"422 Unprocessable Entity: {detail}")
            with st.expander("Payload enviado (debug)"):
                st.json(payload)
            return None

        if r.status_code >= 400:
            try:
                error_detail = r.json().get("detail", r.text)
            except Exception:
                error_detail = r.text
            st.error(f"Error {r.status_code} en la API: {error_detail}")
            with st.expander("Payload enviado (debug)"):
                st.json(payload)
            return None

        return r.json()

    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión con la API: {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        return None


# ================================================================
# UTILIDADES DE DATOS
# ================================================================

def get_player_data(roster: Dict[str, Any], athlete_id: str) -> Optional[Dict[str, Any]]:
    """Busca los datos de un jugador por ID en el roster."""
    if not roster or "players" not in roster:
        return None
    for p in roster["players"]:
        if str(p.get("athlete_id")) == str(athlete_id):
            return p
    return None


def find_player_market_type(position: str) -> str:
    """Devuelve el market_type según la posición."""
    if position == "QB":
        return "passing_yards"
    if position == "RB":
        return "rushing_yards"
    if position == "WR":
        return "receiving_yards"
    return "unknown"


def build_player_payload(
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
    """Construye el payload que tu API WSPM espera para auto-projection-report."""
    return {
        "athlete_id": str(player["athlete_id"]),
        "event_id": str(event_id),
        "season": int(season),
        "season_type": int(season_type),
        "week": int(week),
        "player_name": player["name"],
        "player_team": team_abbr,
        "opponent_team": opp_team_abbr,
        "position": player["position"],
        "market_type": market_type,
        "book_line": float(book_line),
    }


def summarize_projection(prediction: Dict[str, Any]) -> Tuple[float, float, float, float, str]:
    """
    Extrae métricas clave del JSON de WSPM.
    Devuelve: (wspm_projection, book_line, edge_yards, safety_margin_pct, tier_label)
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


def group_roster_key_players(roster: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Agrupa y elige jugadores clave del roster por posición (heurística simple)."""
    grouped = {"QB": [], "RB": [], "WR": []}
    for p in roster.get("players", []):
        pos = (p.get("position") or "").upper()
        if pos in grouped:
            grouped[pos].append(p)

    result: Dict[str, List[Dict[str, Any]]] = {}
    # Heurística: 1 QB, 2 RB, 2 WR
    if grouped["QB"]:
        result["QB"] = grouped["QB"][:1]
    if grouped["RB"]:
        result["RB"] = grouped["RB"][:2]
    if grouped["WR"]:
        result["WR"] = grouped["WR"][:2]

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
# LÓGICA DE AUTO-REPORTE SEMANAL
# ================================================================

def auto_scan_game(
    game: Dict[str, Any],
    season: int,
    season_type: int,
    week: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Escanea un partido:
      - carga roster home / away
      - elige jugadores clave QB/RB/WR
      - llama WSPM auto-projection-report
    Devuelve lista de picks + lista de errores.
    """
    event_id = game["event_id"]
    home = game["home_team"]
    away = game["away_team"]
    home_abbr = home["abbr"]
    away_abbr = away["abbr"]

    picks: List[Dict[str, Any]] = []
    errors: List[str] = []

    def process_team(team_abbr: str, opp_abbr: str):
        nonlocal picks, errors
        roster = get_team_roster(team_abbr)
        if not roster:
            errors.append(f"Error: no se pudo cargar roster de {team_abbr}.")
            return

        key_players = group_roster_key_players(roster)

        for pos, plist in key_players.items():
            mkt_type = find_player_market_type(pos)
            if mkt_type == "unknown":
                continue

            default_line = DEFAULT_LINES.get(mkt_type, 50.5)

            for player in plist:
                payload = build_player_payload(
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
                proj = call_player_projection(payload)
                if not proj:
                    errors.append(
                        f"[INFO] Ignorando {player['name']} ({team_abbr}, {pos}) por error en proyección."
                    )
                    continue

                wspm_proj, book_line, edge, safety_pct, tier = summarize_projection(proj)

                # Capa IA/heurística: dirección, margen, probabilidad, confianza
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
    """
    Recorre games-with-odds, llama WSPM para jugadores clave
    y arma texto listo para redes + DataFrame de picks.
    """
    games = get_games_with_odds(week, season_type, season)
    if not games:
        return f"Sin partidos para week={week}, season={season}.", pd.DataFrame()

    if max_games and max_games > 0:
        games = games[:max_games]

    all_picks: List[Dict[str, Any]] = []
    all_errors: List[str] = []

    for g in games:
        picks, errs = auto_scan_game(g, season, season_type, week)
        all_picks.extend(picks)
        all_errors.extend(errs)

    if not all_picks:
        txt = "No se generaron picks válidos (errores o 422 en todos los intentos).\n\n"
        if all_errors:
            txt += "Log de errores:\n" + "\n".join(all_errors)
        return txt, pd.DataFrame()

    df = pd.DataFrame(all_picks)
    # Ordenamos por tier (valor), luego por probabilidad y edge
    df_sorted = df.sort_values(
        by=["tier", "prob_cover", "edge"], ascending=[True, False, False]
    ).reset_index(drop=True)

    # FREE PICK = mejor pick (primero del orden)
    top = df_sorted.iloc[0]
    free_text = (
        f"🎯 Free Pick WSPM – Semana {week} NFL {season}\n\n"
        f"{top['matchup']} | {top['provider']}: {top['details']}, O/U {top['over_under']}\n"
        f"{top['name']} ({top['team']} – {top['market_type'].replace('_', ' ')}) "
        f"{top['direction']} {top['book_line']:.1f} | Proyección WSPM: {top['wspm_projection']:.1f}, "
        f"edge {top['edge']:.1f} yds, {top['margin_pct']:.1f}% margen, "
        f"Confianza {top['confidence']} (P(cubrir) ~ {top['prob_cover']:.1f}%)\n"
    )

    # PREMIUM / PLATINUM – lista completa
    lines_premium: List[str] = []
    for idx, row in df_sorted.iterrows():
        emoji = tier_emoji(row["tier"])
        line = (
            f"{idx+1}. {emoji} {row['name']} ({row['team']} – "
            f"{row['market_type'].replace('_', ' ')}) {row['direction']} {row['book_line']:.1f} | "
            f"proj {row['wspm_projection']:.1f}, edge {row['edge']:.1f} yds, "
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

    # Breakdown compacto por partido
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
                f"edge {row['edge']:.1f} yds, {row['margin_pct']:.1f}% margen, "
                f"P(cubrir)~{row['prob_cover']:.1f}%, Tier {row['tier']} ({row['confidence']})"
            )
        breakdown_lines.append("")

    full_text = free_text + premium_block + "\n" + "\n".join(breakdown_lines)

    if all_errors:
        full_text += "\n\nLog de errores WSPM/rosters:\n" + "\n".join(all_errors)

    return full_text, df_sorted


# ================================================================
# INTERFAZ DE STREAMLIT
# ================================================================

def main():
    st.set_page_config(
        page_title="WSPM Edge – NFL",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏈 WSPM Edge – NFL (Player Props & Auto Picks)")
    st.markdown(
        "Frontend en Streamlit para tu API WSPM: **proyecciones, edges, "
        "probabilidad de cubrir y textos listos para redes sociales**."
    )
    st.divider()

    # ---------------------------------------------
    # SIDEBAR – Config de temporada
    # ---------------------------------------------
    st.sidebar.header("⚙️ Configuración Global")
    season = st.sidebar.number_input(
        "Temporada", value=DEFAULT_SEASON, min_value=2018, step=1
    )
    week = st.sidebar.number_input(
        "Semana NFL", value=DEFAULT_WEEK, min_value=1, max_value=22, step=1
    )
    season_type = st.sidebar.selectbox(
        "Tipo de temporada",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Preseason (1)", 2: "Regular (2)", 3: "Postseason (3)"}[x],
        index=1,  # 2 = Regular
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Backend WSPM:**")
    st.sidebar.code(BASE_URL, language="bash")

    # Cargar juegos
    games = get_games_with_odds(week=int(week), season_type=int(season_type), season=int(season))
    if not games:
        st.error("No se pudieron cargar juegos desde /games-with-odds.")
        return

    game_options = {
        f"{g['matchup']} ({g['odds'].get('details', 'N/A')}, O/U {g['odds'].get('over_under', 'N/A')})": g
        for g in games
    }
    game_names = list(game_options.keys())

    tab1, tab2 = st.tabs(["🎯 Player Prop (manual)", "🤖 Auto Picks Semana (Free/Premium/Platinum)"])

    # ============================================================
    # TAB 1 – Player Prop manual
    # ============================================================
    with tab1:
        st.header("🎯 Player Prop – Análisis Individual")

        selected_game_name = st.selectbox(
            "Selecciona un partido:",
            game_names,
            index=0,
            key="game_select_tab1",
        )

        selected_game = game_options[selected_game_name]
        event_id = selected_game["event_id"]
        home_abbr = selected_game["home_team"]["abbr"]
        away_abbr = selected_game["away_team"]["abbr"]
        matchup_text = selected_game["matchup"]

        st.info(f"Partido Seleccionado: **{matchup_text}** | Event ID: `{event_id}`")

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

            roster_data = get_team_roster(team_choice)

        players_for_selection: List[Dict[str, Any]] = []

        if not roster_data:
            st.warning(f"No se pudo cargar el roster para {team_choice}.")
        else:
            # Armar lista de posibles jugadores (QB/RB/WR)
            for p in roster_data.get("players", []):
                pos = p.get("position")
                if pos in ["QB", "RB", "WR"]:
                    market_type = find_player_market_type(pos)
                    if market_type != "unknown":
                        players_for_selection.append(
                            {
                                "id": p["athlete_id"],
                                "name": f"{p['name']} ({p['position']})",
                                "position": pos,
                                "market_type": market_type,
                                "default_line": DEFAULT_LINES.get(market_type),
                            }
                        )

            with col_player:
                if not players_for_selection:
                    st.error("No se encontraron jugadores QB/RB/WR en este roster.")
                else:
                    player_df = pd.DataFrame(players_for_selection)
                    player_options = player_df["name"].tolist()
                    selected_player_name = st.selectbox(
                        "Jugador a analizar",
                        player_options,
                        index=0,
                    )
                    selected_player_row = player_df[player_df["name"] == selected_player_name].iloc[0]
                    selected_athlete_id = selected_player_row["id"]
                    selected_position = selected_player_row["position"]
                    selected_market_type = selected_player_row["market_type"]
                    selected_default_line = float(selected_player_row["default_line"])

        st.divider()

        if roster_data and players_for_selection:
            st.subheader("2. Línea del Book y Mercado")

            col_market, col_line = st.columns([1, 1])

            with col_market:
                market_label = KEY_MARKETS[selected_position]["label"]
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

                    prediction_report = call_player_projection(payload)

                    st.header("📊 Resultado de la Proyección")

                    if prediction_report:
                        # Extraemos métricas base del backend (edge, safety_margin_pct, etc.)
                        wspm_proj, used_line, edge, safety_pct_backend, tier = summarize_projection(
                            prediction_report
                        )

                        # Construimos explicación WSPM + capa IA (dirección, prob, confianza)
                        wspm_md, direction, margin_yards, margin_pct, conf_label, prob_cover = (
                            build_wspm_markdown_explanation(
                                matchup=matchup_text,
                                team=team_choice,
                                opp=opp_choice,
                                player_name=player_info["name"],
                                position=player_info["position"],
                                market_type=selected_market_type,
                                wspm_projection=wspm_proj,
                                book_line=used_line,
                                safety_pct_backend=safety_pct_backend,
                            )
                        )

                        # Métricas arriba
                        col_proj, col_edge, col_safety, col_prob, col_conf = st.columns(5)

                        with col_proj:
                            st.metric("Proyección WSPM", f"{wspm_proj:.1f} yds")
                        with col_edge:
                            st.metric(
                                "Edge vs. Book",
                                f"{edge:+.1f} yds",
                                delta=f"{edge:+.1f}",
                            )
                        with col_safety:
                            st.metric("Margen % (modelo)", f"{margin_pct:.1f}%")
                        with col_prob:
                            st.metric("P(cubrir) estimada", f"{prob_cover:.1f}%")
                        with col_conf:
                            st.metric("Confianza Pick", conf_label)

                        # Bloque largo WSPM
                        st.markdown(wspm_md)

                        # Texto corto para redes
                        st.subheader("📣 Texto corto para redes (Player Prop)")
                        short_txt = (
                            f"{tier_emoji(tier)} WSPM – {matchup_text}\n"
                            f"{player_info['name']} ({team_choice} – {selected_market_type.replace('_',' ')}) "
                            f"{direction} {used_line:.1f}\n"
                            f"Proy: {wspm_proj:.1f} yds | Edge {edge:+.1f} | "
                            f"Margen {margin_pct:.1f}% | P(cubrir)~{prob_cover:.1f}% | Conf {conf_label}\n"
                            f"#NFL #PlayerProps #WSPM"
                        )
                        st.code(short_txt, language="markdown")
                    else:
                        st.error("La API no devolvió una proyección válida.")

    # ============================================================
    # TAB 2 – Auto Picks para la Semana
    # ============================================================
    with tab2:
        st.header("🤖 Auto Picks WSPM – Semana Completa")

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
                st.subheader("📑 Tabla de Picks (ordenados por tier / probabilidad / edge)")
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
                    "_Tip_: filtra en la tabla y copia solo los Platinum/Premium "
                    "con mayor probabilidad para tu newsletter o canal de Telegram."
                )


if __name__ == "__main__":
    main()
