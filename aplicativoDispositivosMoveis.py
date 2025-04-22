import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback, no_update, dash_table
import dash_bootstrap_components as dbc
import pathlib
from scipy import stats
import time
from dash.dependencies import State, ALL
import logging
from flask_caching import Cache

# ========== CONFIGURAÇÃO INICIAL ==========
app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           meta_tags=[
               {'name': 'viewport',
                'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5'}
           ])
app.title = "Painel de Segurança Pública (2001-2023)"
server = app.server

# Configuração de cache
cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 3600  # 1 hora
})

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== FUNÇÕES AUXILIARES ==========
@cache.memoize(timeout=3600)
def carregar_dados():
    """Carrega e processa os dados do arquivo Excel com tratamento robusto de erros."""
    try:
        start_time = time.time()
        meses_esperados = [
            'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
            'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'
        ]

        # Caminho do arquivo
        file_path = pathlib.Path(__file__).parent / "Dados Segurança Publica 2001 a 2023 consolidado.xlsx"

        if not file_path.exists():
            available_files = [f.name for f in pathlib.Path(__file__).parent.iterdir()]
            logger.error(f"Arquivo não encontrado. Arquivos disponíveis: {available_files}")
            return None

        logger.info(f"Carregando arquivo: {file_path}")
        xl = pd.ExcelFile(file_path, engine='openpyxl')
        dfs = []
        anos_com_erro = []

        for ano in range(2001, 2024):
            sheet_name = str(ano)
            if sheet_name not in xl.sheet_names:
                logger.warning(f"Planilha para o ano {ano} não encontrada")
                anos_com_erro.append(ano)
                continue

            try:
                # Lê a planilha como texto inicialmente
                df = pd.read_excel(
                    xl,
                    sheet_name=sheet_name,
                    header=None,
                    dtype=str,
                    engine='openpyxl',
                    na_values=['-', 'NA', 'NULL', 'NaN', '']
                ).dropna(how='all').reset_index(drop=True)

                if df.empty:
                    logger.warning(f"Planilha {ano} está vazia")
                    anos_com_erro.append(ano)
                    continue

                # Encontra a linha do cabeçalho
                mask = df.apply(lambda col: col.str.contains('Natureza', case=False, na=False)).any(axis=1)
                if not mask.any():
                    logger.warning(f"Cabeçalho não encontrado na planilha {ano}")
                    anos_com_erro.append(ano)
                    continue

                header_row = mask.idxmax()
                df.columns = df.iloc[header_row].str.strip()
                df = df.iloc[header_row + 1:].dropna(how='all', axis=1)

                # Verifica se temos colunas suficientes
                if df.shape[1] < 13:  # Natureza + 12 meses
                    logger.warning(f"Estrutura inválida na planilha {ano} - colunas insuficientes")
                    anos_com_erro.append(ano)
                    continue

                # Processa os meses (colunas 1 a 12)
                meses_df = df.iloc[:, 1:13].copy()
                meses_df.columns = meses_esperados

                # Combina com a coluna Natureza
                df_final = pd.concat([
                    df.iloc[:, 0].str.strip().rename('Natureza'),
                    meses_df
                ], axis=1)

                # Converte valores numéricos
                for mes in meses_esperados:
                    df_final[mes] = (
                        df_final[mes]
                        .astype(str)
                        .str.replace(r'[^\d,]', '', regex=True)
                        .str.replace(',', '.')
                        .replace(['', 'nan', 'None'], pd.NA)
                        .pipe(pd.to_numeric, errors='coerce')
                    )

                # Remove linhas inválidas
                df_final = df_final.dropna(how='all', subset=meses_esperados)

                if df_final.empty:
                    logger.warning(f"Nenhum dado válido para o ano {ano}")
                    anos_com_erro.append(ano)
                    continue

                df_final['Ano'] = ano
                dfs.append(df_final)

            except Exception as e:
                logger.error(f"Erro no processamento do ano {ano}: {str(e)}", exc_info=True)
                anos_com_erro.append(ano)
                continue

        if not dfs:
            logger.error("Nenhuma planilha válida foi carregada")
            return None

        # Consolida todos os dados
        df_completo = pd.concat(dfs, ignore_index=True)

        # Transforma para formato longo
        df_long = df_completo.melt(
            id_vars=['Ano', 'Natureza'],
            value_vars=meses_esperados,
            var_name='Mês',
            value_name='Ocorrências'
        ).dropna(subset=['Ocorrências'])

        # Cria coluna de data
        mes_para_num = {mes: i + 1 for i, mes in enumerate(meses_esperados)}
        df_long['Data'] = pd.to_datetime(
            df_long['Ano'].astype(str) + '-' +
            df_long['Mês'].map(mes_para_num).astype(str) + '-01',
            errors='coerce'
        )

        df_long = df_long.dropna(subset=['Data'])

        logger.info(f"Dados carregados com sucesso em {time.time() - start_time:.2f} segundos")
        logger.info(f"Anos com problemas: {anos_com_erro}")

        return df_long.sort_values(['Ano', 'Mês'])

    except Exception as e:
        logger.critical(f"ERRO CRÍTICO: {str(e)}", exc_info=True)
        return None


def calcular_regressao_media(df, crime_selecionado):
    """Calcula a regressão linear e média móvel para um crime específico."""
    dados = df[df['Natureza'] == crime_selecionado]
    dados_anuais = dados.groupby('Ano')['Ocorrências'].sum().reset_index()

    x = dados_anuais['Ano'].values
    y = dados_anuais['Ocorrências'].values

    if len(x) < 2:
        return None

    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    linha_regressao = intercept + slope * x
    media_movel = pd.Series(y).rolling(window=3, center=True, min_periods=1).mean().values

    return {
        'Anos': x,
        'Ocorrências': y,
        'Regressão': linha_regressao,
        'Média Móvel': media_movel,
        'Coeficiente': slope,
        'Intercepto': intercept,
        'R²': r_value ** 2
    }


def criar_card_estatistica(titulo, valor, cor="primary"):
    """Cria um componente de card para estatísticas."""
    return dbc.Card(
        dbc.CardBody([
            html.H6(titulo, className="card-subtitle"),
            html.P(valor, className=f"card-text text-{cor} fw-bold")
        ]),
        className="text-center shadow-sm mb-3"
    )


# ========== LAYOUT RESPONSIVO ==========
def criar_layout():
    return dbc.Container(
        [
            dcc.Loading(
                id="loading",
                type="default",
                children=[
                    html.Div(id="loading-output"),
                    dcc.Store(id='dados-store'),
                    dcc.Interval(id='interval-trigger', interval=1000, max_intervals=1),
                    dcc.Download(id="download-data"),
                ]
            ),

            dbc.Row(
                dbc.Col(
                    [
                        html.H1(
                            "Painel Criminalidade",
                            className="text-center my-2",
                            style={
                                'fontSize': 'clamp(1.5rem, 4vw, 2rem)',
                                'marginBottom': '1rem'
                            }
                        ),

                        dbc.Accordion(
                            [
                                dbc.AccordionItem(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Label("Tipo de Crime:", className="fw-bold mb-1"),
                                                    width=12
                                                ),
                                                dbc.Col(
                                                    dcc.Dropdown(
                                                        id='crime-dropdown',
                                                        placeholder="Selecione...",
                                                        clearable=False,
                                                        style={'minHeight': '40px'}
                                                    ),
                                                    width=12
                                                )
                                            ],
                                            className="mb-3"
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Label("Período:", className="fw-bold mb-1"),
                                                    width=12
                                                ),
                                                dbc.Col(
                                                    dcc.RangeSlider(
                                                        id='ano-slider',
                                                        min=2001,
                                                        max=2023,
                                                        value=[2018, 2023],
                                                        marks={
                                                            str(ano): {'label': str(ano), 'style': {'fontSize': '10px'}}
                                                            for ano in range(2001, 2024, 3)
                                                        },
                                                        tooltip={"placement": "bottom"}
                                                    ),
                                                    width=12
                                                )
                                            ],
                                            className="mb-3"
                                        )
                                    ],
                                    title="Filtros",
                                    item_id="filters"
                                )
                            ],
                            start_collapsed=True,
                            flush=True,
                            className="mb-3"
                        ),

                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    [
                                        dbc.Row(
                                            dbc.Col(
                                                dcc.Graph(
                                                    id='serie-temporal',
                                                    config={'displayModeBar': True},
                                                    style={'height': '40vh'}
                                                ),
                                                width=12
                                            ),
                                            className="mb-3"
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id='grafico-sazonal',
                                                        config={'displayModeBar': False},
                                                        style={'height': '35vh'}
                                                    ),
                                                    xs=12, md=6, className="mb-3"
                                                ),
                                                dbc.Col(
                                                    dcc.Graph(
                                                        id='grafico-anual',
                                                        config={'displayModeBar': False},
                                                        style={'height': '35vh'}
                                                    ),
                                                    xs=12, md=6, className="mb-3"
                                                )
                                            ]
                                        )
                                    ],
                                    label="Visualizações",
                                    tab_id="visualizations"
                                ),
                                dbc.Tab(
                                    dbc.Row(
                                        dbc.Col(
                                            dcc.Graph(
                                                id='grafico-regressao',
                                                config={'displayModeBar': False},
                                                style={'height': '50vh'}
                                            ),
                                            width=12
                                        )
                                    ),
                                    label="Tendência",
                                    tab_id="trend"
                                ),
                                dbc.Tab(
                                    [
                                        dbc.Row(
                                            dbc.Col(
                                                dash_table.DataTable(
                                                    id='tabela-dados',
                                                    page_size=10,
                                                    style_table={
                                                        'overflowX': 'auto',
                                                        'maxWidth': '100%'
                                                    },
                                                    style_cell={
                                                        'textAlign': 'left',
                                                        'padding': '8px',
                                                        'minWidth': '80px',
                                                        'fontSize': 'clamp(10px, 1.5vw, 12px)',
                                                        'whiteSpace': 'normal',
                                                        'height': 'auto'
                                                    },
                                                    style_header={
                                                        'backgroundColor': '#f8f9fa',
                                                        'fontWeight': 'bold'
                                                    }
                                                ),
                                                width=12
                                            ),
                                            className="mb-3"
                                        ),
                                        dbc.Row(
                                            dbc.Col(
                                                html.Div(id='estatisticas-resumo'),
                                                width=12
                                            )
                                        )
                                    ],
                                    label="Dados",
                                    tab_id="data"
                                )
                            ],
                            id="main-tabs",
                            active_tab="visualizations"
                        )
                    ],
                    xs=12, lg=10, xl=8
                ),
                justify="center"
            ),

            html.Div(
                dbc.Button(
                    html.I(className="fas fa-file-export"),
                    id="btn-export",
                    color="primary",
                    className="position-fixed",
                    style={
                        'bottom': '20px',
                        'right': '20px',
                        'zIndex': '1000',
                        'width': '50px',
                        'height': '50px',
                        'borderRadius': '50%',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center'
                    }
                )
            ),

            html.Div([
                html.Link(
                    rel='stylesheet',
                    href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
                ),
                dcc.Markdown(
                    '''
                    <style>
                        @media (max-width: 767px) {
                            .dash-graph {
                                height: 45vh !important;
                            }
                            .dash-table-container {
                                font-size: 11px !important;
                            }
                        }
                        @media (min-width: 768px) and (max-width: 1024px) {
                            .dash-graph {
                                height: 50vh !important;
                            }
                        }
                    </style>
                    ''',
                    dangerously_allow_html=True
                )
            ])
        ],
        fluid=True,
        className="px-2 py-1",
        style={'maxWidth': '1200px'}
    )


# ========== CALLBACKS ==========
def configurar_callbacks():
    @app.callback(
        [Output('dados-store', 'data'),
         Output('crime-dropdown', 'options')],
        Input('interval-trigger', 'n_intervals')
    )
    def carregar_dados_inicial(_):
        """Carrega os dados e atualiza o dropdown inicial."""
        dados = carregar_dados()
        if dados is None:
            return no_update, []

        # Converte para formato adequado para o dcc.Store
        dados_dict = dados.to_dict('records')

        # Opções para o dropdown
        crimes = dados['Natureza'].unique()
        opcoes = [{'label': crime, 'value': crime} for crime in sorted(crimes)]

        return dados_dict, opcoes

    @app.callback(
        Output('crime-dropdown', 'options'),
        Input('ano-slider', 'value'),
        State('dados-store', 'data')
    )
    def filtrar_crimes_por_ano(anos_selecionados, dados):
        """Filtra os crimes disponíveis com base no período selecionado."""
        if not dados or not anos_selecionados:
            return []

        df = pd.DataFrame.from_records(dados)
        crimes_filtrados = df[
            df['Ano'].between(anos_selecionados[0], anos_selecionados[1])
        ]['Natureza'].unique()

        return [{'label': crime, 'value': crime} for crime in sorted(crimes_filtrados)]

    @app.callback(
        [Output('serie-temporal', 'figure'),
         Output('grafico-sazonal', 'figure'),
         Output('grafico-anual', 'figure'),
         Output('grafico-regressao', 'figure'),
         Output('estatisticas-resumo', 'children'),
         Output('tabela-dados', 'data'),
         Output('tabela-dados', 'columns')],
        [Input('crime-dropdown', 'value'),
         Input('ano-slider', 'value')],
        State('dados-store', 'data')
    )
    def atualizar_visualizacoes(crime_selecionado, anos_selecionados, dados):
        """Atualiza todos os gráficos e componentes com base nos filtros."""
        # Valores padrão para primeira execução
        if not crime_selecionado:
            crime_selecionado = 'HOMICÍDIO DOLOSO (2)'
        if not anos_selecionados:
            anos_selecionados = [2018, 2023]

        if not dados:
            empty_fig = px.scatter(title="Dados não carregados")
            empty_fig.update_layout(template='plotly_white')
            columns = [{'name': col, 'id': col} for col in ['Ano', 'Mês', 'Ocorrências']]
            return empty_fig, empty_fig, empty_fig, empty_fig, "Dados não carregados", [], columns

        df = pd.DataFrame.from_records(dados)
        df_filtrado = df[
            (df['Natureza'] == crime_selecionado) &
            (df['Ano'].between(anos_selecionados[0], anos_selecionados[1]))
            ]

        if df_filtrado.empty:
            empty_fig = px.scatter(title="Nenhum dado encontrado")
            empty_fig.update_layout(template='plotly_white')
            columns = [{'name': col, 'id': col} for col in ['Ano', 'Mês', 'Ocorrências']]
            return empty_fig, empty_fig, empty_fig, empty_fig, "Nenhum dado encontrado", [], columns

        # 1. Gráfico de Série Temporal
        fig_serie = px.line(
            df_filtrado,
            x='Data',
            y='Ocorrências',
            title=f'Série Temporal: {crime_selecionado}',
            labels={'Ocorrências': 'Número de Ocorrências'},
            template='plotly_white'
        )
        fig_serie.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='x unified'
        )

        # 2. Gráfico Sazonal
        sazonal = df_filtrado.groupby('Mês', as_index=False)['Ocorrências'].mean()
        fig_sazonal = px.bar(
            sazonal,
            x='Mês',
            y='Ocorrências',
            title=f'Padrão Sazonal: {crime_selecionado}',
            color='Ocorrências',
            color_continuous_scale='Blues',
            template='plotly_white'
        )
        fig_sazonal.update_layout(coloraxis_showscale=False)

        # 3. Gráfico Anual
        anual = df_filtrado.groupby('Ano', as_index=False)['Ocorrências'].sum()
        fig_anual = px.bar(
            anual,
            x='Ano',
            y='Ocorrências',
            title=f'Evolução Anual: {crime_selecionado}',
            text='Ocorrências',
            template='plotly_white'
        )
        fig_anual.update_traces(
            texttemplate='%{text:,.0f}',
            textposition='outside',
            marker_color='#3498db'
        )

        # 4. Gráfico de Regressão
        regressao = calcular_regressao_media(df_filtrado, crime_selecionado)
        if regressao:
            fig_regressao = px.scatter(
                x=regressao['Anos'],
                y=regressao['Ocorrências'],
                title=f'Regressão à Média (R²={regressao["R²"]:.2f})',
                labels={'x': 'Ano', 'y': 'Ocorrências'},
                template='plotly_white'
            )
            fig_regressao.add_scatter(
                x=regressao['Anos'],
                y=regressao['Regressão'],
                mode='lines',
                name='Tendência Linear',
                line=dict(color='red', width=3)
            )
            fig_regressao.add_scatter(
                x=regressao['Anos'],
                y=regressao['Média Móvel'],
                mode='lines',
                name='Média Móvel (3 anos)',
                line=dict(color='green', width=2, dash='dot')
            )
        else:
            fig_regressao = px.scatter(
                title="Dados insuficientes para regressão",
                template='plotly_white'
            )

        # 5. Estatísticas
        stats = df_filtrado.groupby('Ano')['Ocorrências'].sum().describe()
        tem_tendencia = regressao and regressao['Coeficiente'] > 0 if regressao else False

        stats_html = [
            html.H4(crime_selecionado, className="card-title mb-3"),
            dbc.Row([
                dbc.Col(criar_card_estatistica(
                    "Total de Ocorrências",
                    f"{df_filtrado['Ocorrências'].sum():,.0f}",
                    "primary"
                ), md=3),
                dbc.Col(criar_card_estatistica(
                    "Média Anual",
                    f"{stats['mean']:,.0f}",
                    "info"
                ), md=3),
                dbc.Col(criar_card_estatistica(
                    "Tendência Anual",
                    f"{'↑' if tem_tendencia else '↓'} {abs(regressao['Coeficiente']):.1f} casos/ano" if regressao else "N/A",
                    "success" if tem_tendencia else "danger"
                ), md=3),
                dbc.Col(criar_card_estatistica(
                    "Qualidade do Ajuste (R²)",
                    f"{regressao['R²']:.2f}" if regressao else "N/A",
                    "warning"
                ), md=3)
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col(criar_card_estatistica(
                    "Mês com Mais Ocorrências",
                    sazonal.loc[sazonal['Ocorrências'].idxmax(), 'Mês'],
                    "danger"
                ), md=4),
                dbc.Col(criar_card_estatistica(
                    "Mês com Menos Ocorrências",
                    sazonal.loc[sazonal['Ocorrências'].idxmin(), 'Mês'],
                    "success"
                ), md=4),
                dbc.Col(criar_card_estatistica(
                    "Ano com Mais Ocorrências",
                    str(anual.loc[anual['Ocorrências'].idxmax(), 'Ano']),
                    "danger"
                ), md=4)
            ])
        ]

        # 6. Tabela de dados
        tabela_data = df_filtrado[['Ano', 'Mês', 'Ocorrências']].to_dict('records')
        tabela_columns = [{'name': col, 'id': col} for col in ['Ano', 'Mês', 'Ocorrências']]

        return fig_serie, fig_sazonal, fig_anual, fig_regressao, stats_html, tabela_data, tabela_columns

    @app.callback(
        Output("download-data", "data"),
        Input("btn-export", "n_clicks"),
        [State('crime-dropdown', 'value'),
         State('ano-slider', 'value'),
         State('dados-store', 'data')],
        prevent_initial_call=True
    )
    def exportar_dados(n_clicks, crime, anos, dados):
        """Exporta os dados filtrados para um arquivo CSV."""
        if n_clicks is None or not crime or not dados:
            return no_update

        df = pd.DataFrame.from_records(dados)
        filtered = df[(df['Natureza'] == crime) &
                      (df['Ano'].between(anos[0], anos[1]))]

        if filtered.empty:
            return no_update

        return {
            'content': filtered.to_csv(index=False),
            'filename': f"dados_criminalidade_{crime}_{anos[0]}-{anos[1]}.csv"
        }

    @app.callback(
        Output("loading-output", "children"),
        Input('dados-store', 'data')
    )
    def verificar_dados(dados):
        if dados:
            df = pd.DataFrame.from_records(dados)
            return f"Dados carregados: {len(df)} registros"
        return "Carregando dados..."


# ========== CONFIGURAÇÃO FINAL ==========
app.layout = criar_layout()
configurar_callbacks()

if __name__ == '__main__':
    app.run(port=8080, debug=True)