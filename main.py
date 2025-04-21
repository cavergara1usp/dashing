import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback, no_update, dash_table
import dash_bootstrap_components as dbc
from datetime import datetime
import os
import pathlib
from scipy import stats
import time
from dash.dependencies import State, ALL
import logging
from flask_caching import Cache

# ========== CONFIGURAÇÃO INICIAL ==========
app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])
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


# ========== VERIFICAÇÃO INICIAL DO ARQUIVO ==========
def verificar_arquivo():
    """Verifica se o arquivo de dados existe em locais possíveis."""
    caminhos_tentados = [
        pathlib.Path(__file__).parent / "Dados Segurança Publica 2001 a 2023 consolidado.xlsx",
        pathlib.Path("Dados Segurança Publica 2001 a 2023 consolidado.xlsx"),
        pathlib.Path("pages.main") / "Dados Segurança Publica 2001 a 2023 consolidado.xlsx"
    ]

    for caminho in caminhos_tentados:
        if caminho.exists():
            logger.info(f"Arquivo encontrado em: {caminho.resolve()}")
            return caminho

    logger.error("Arquivo não encontrado em nenhum dos caminhos tentados")
    return None


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

        # Verifica o arquivo em vários locais possíveis
        file_path = verificar_arquivo()
        if file_path is None:
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
                anos_com_erro.append(ano)
                continue

            try:
                df = pd.read_excel(
                    xl,
                    sheet_name=sheet_name,
                    header=None,
                    dtype=str,
                    engine='openpyxl'
                ).dropna(how='all')

                if df.empty:
                    anos_com_erro.append(ano)
                    continue

                # Encontra o cabeçalho
                mask = df[0].str.contains('Natureza', na=False, case=False)
                if not mask.any():
                    logger.warning(f"Cabeçalho não encontrado na planilha {ano}")
                    anos_com_erro.append(ano)
                    continue

                natureza_idx = mask.idxmax()
                df.columns = df.iloc[natureza_idx]
                df = df.iloc[natureza_idx + 1:].dropna(how='all', axis=1)

                # Processa os meses
                meses_df = df.iloc[:, 1:13].copy()
                meses_df.columns = meses_esperados
                df_final = pd.concat([
                    df['Natureza'].str.strip(),
                    meses_df
                ], axis=1)

                # Converte valores numéricos
                for mes in meses_esperados:
                    df_final[mes] = (
                        df_final[mes]
                        .astype(str)
                        .str.replace(r'[^\d,]', '', regex=True)
                        .str.replace(',', '.')
                        .pipe(pd.to_numeric, errors='coerce')
                    )

                df_final['Ano'] = ano
                dfs.append(df_final)

            except Exception as e:
                logger.error(f"Erro no ano {ano}: {str(e)}", exc_info=True)
                anos_com_erro.append(ano)
                continue

        if not dfs:
            raise ValueError("Nenhuma planilha válida foi carregada")

        # Consolidação final
        df_completo = pd.concat(dfs, ignore_index=True).convert_dtypes()

        # Transformação para formato longo
        df_long = df_completo.melt(
            id_vars=['Ano', 'Natureza'],
            value_vars=meses_esperados,
            var_name='Mês',
            value_name='Ocorrências'
        ).dropna()

        # Criação de datas
        mes_para_num = {mes: i + 1 for i, mes in enumerate(meses_esperados)}
        df_long['Data'] = pd.to_datetime(
            df_long['Ano'].astype(str) + '-' +
            df_long['Mês'].map(mes_para_num).astype(str) + '-01'
        )

        logger.info(f"Dados carregados em {time.time() - start_time:.2f} segundos")
        logger.info(f"Anos com problemas: {anos_com_erro}")
        return df_long.sort_values('Data')

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


# ========== LAYOUT DO PAINEL ==========
def criar_layout():
    return dbc.Container([
        dcc.Store(id='dados-store'),
        dcc.Interval(id='interval-trigger', interval=1000, max_intervals=1),
        dcc.Download(id="download-data"),

        dcc.Loading(
            id="loading",
            type="circle",
            children=[
                dbc.Row([
                    dbc.Col(html.H1("Painel de Criminalidade", className="text-center my-4"), width=12),
                    dbc.Col([
                        dbc.Alert(
                            "Carregando dados...",
                            id="loading-message",
                            color="info",
                            dismissable=True,
                            is_open=False
                        )
                    ], width=12)
                ]),

                # Controles de Filtro
                dbc.Row([
                    dbc.Col([
                        html.Label("Selecione o Tipo de Crime:", className="fw-bold mb-2"),
                        dcc.Dropdown(
                            id='crime-dropdown',
                            placeholder="Selecione um crime...",
                            disabled=True
                        )
                    ], md=6),

                    dbc.Col([
                        html.Label("Selecione o Período:", className="fw-bold mb-2"),
                        dcc.RangeSlider(
                            id='ano-slider',
                            min=2001,
                            max=2023,
                            value=[2018, 2023],
                            marks={str(ano): {'label': str(ano)} for ano in range(2001, 2024, 2)},
                            step=1,
                            disabled=True,
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], md=6)
                ], className="mb-4"),

                # Botão de Exportação
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            "Exportar Dados Filtrados",
                            id="btn-export",
                            color="secondary",
                            className="me-2",
                            disabled=True
                        ),
                        width=12
                    ),
                    className="mb-3"
                ),

                # Gráfico de Série Temporal
                dbc.Row(
                    dbc.Col(
                        dcc.Graph(id='serie-temporal'),
                        width=12
                    ),
                    className="mb-4"
                ),

                # Gráficos Sazonal e Anual
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(id='grafico-sazonal'),
                        width=6
                    ),
                    dbc.Col(
                        dcc.Graph(id='grafico-anual'),
                        width=6
                    )
                ], className="mb-4"),

                # Gráfico de Regressão à Média
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader("Análise de Tendência", className="fw-bold"),
                            dbc.CardBody([
                                dcc.Graph(id='grafico-regressao'),
                                html.Div(id='regressao-stats', className="mt-3")
                            ])
                        ]),
                        width=12
                    ),
                    className="mb-4"
                ),

                # Estatísticas Resumidas
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader("Análise Estatística", className="fw-bold"),
                            dbc.CardBody(id='estatisticas-resumo')
                        ]),
                        width=12
                    ),
                    className="mb-4"
                ),

                # Tabela de Dados
                dbc.Row(
                    dbc.Col(
                        dash_table.DataTable(
                            id='tabela-dados',
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left',
                                'padding': '8px',
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            },
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                            }
                        ),
                        width=12
                    ),
                    className="mb-4"
                )
            ]
        )
    ], fluid=True, className="py-4")


def configurar_callbacks():
    @app.callback(
        [Output('dados-store', 'data'),
         Output('loading-message', 'is_open'),
         Output('crime-dropdown', 'disabled'),
         Output('ano-slider', 'disabled'),
         Output('btn-export', 'disabled')],
        Input('interval-trigger', 'n_intervals'),
        prevent_initial_call=False
    )
    def carregar_e_habilitar(_):
        """Carrega os dados e habilita os controles do painel."""
        dados = carregar_dados()
        if dados is None:
            return no_update, True, True, True, True

        return dados.to_dict('records'), False, False, False, False

    @app.callback(
        Output('crime-dropdown', 'options'),
        Input('dados-store', 'data'),
        State('ano-slider', 'value')
    )
    def atualizar_dropdown(dados, anos_selecionados):
        """Atualiza as opções do dropdown de crimes com base nos anos selecionados."""
        if dados is None:
            return []

        df = pd.DataFrame.from_records(dados)
        crimes = df[df['Ano'].between(anos_selecionados[0], anos_selecionados[1])]['Natureza'].unique()
        return [{'label': crime, 'value': crime} for crime in sorted(crimes)]

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
    def atualizar_graficos(selected_crime, selected_years, dados):
        """Atualiza todos os gráficos e estatísticas com base nos filtros selecionados."""
        if dados is None or not selected_crime or not selected_years:
            empty = px.scatter(title="Selecione um tipo de crime e período")
            empty.update_layout(template='plotly_white')

            columns = [{'name': col, 'id': col} for col in ['Ano', 'Mês', 'Ocorrências']]
            return empty, empty, empty, empty, "Selecione um tipo de crime", [], columns

        df = pd.DataFrame.from_records(dados)
        filtered = df[(df['Natureza'] == selected_crime) &
                      (df['Ano'].between(selected_years[0], selected_years[1]))]

        if filtered.empty:
            empty = px.scatter(title="Nenhum dado encontrado")
            empty.update_layout(template='plotly_white')

            columns = [{'name': col, 'id': col} for col in ['Ano', 'Mês', 'Ocorrências']]
            return empty, empty, empty, empty, "Nenhum dado encontrado", [], columns

        # 1. Gráfico de Série Temporal
        fig_serie = px.line(
            filtered,
            x='Data',
            y='Ocorrências',
            title=f'Série Temporal: {selected_crime}',
            labels={'Ocorrências': 'Número de Ocorrências'},
            template='plotly_white'
        )
        fig_serie.update_traces(line=dict(width=2.5))
        fig_serie.update_layout(
            hovermode='x unified',
            xaxis_rangeslider_visible=True,
            hoverlabel=dict(bgcolor="white", font_size=12)
        )

        # 2. Gráfico Sazonal
        sazonal = filtered.groupby('Mês', as_index=False)['Ocorrências'].mean()
        fig_sazonal = px.bar(
            sazonal,
            x='Mês',
            y='Ocorrências',
            title=f'Padrão Sazonal: {selected_crime}',
            color='Ocorrências',
            color_continuous_scale='Blues',
            template='plotly_white'
        )
        fig_sazonal.update_layout(coloraxis_showscale=False)

        # 3. Gráfico Anual
        anual = filtered.groupby('Ano', as_index=False)['Ocorrências'].sum()
        fig_anual = px.bar(
            anual,
            x='Ano',
            y='Ocorrências',
            title=f'Evolução Anual: {selected_crime}',
            text='Ocorrências',
            template='plotly_white'
        )
        fig_anual.update_traces(
            texttemplate='%{text:,.0f}',
            textposition='outside',
            marker_color='#3498db'
        )

        # 4. Gráfico de Regressão
        regressao = calcular_regressao_media(filtered, selected_crime)
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
        stats = filtered.groupby('Ano')['Ocorrências'].sum().describe()
        tem_tendencia = regressao and regressao['Coeficiente'] > 0 if regressao else False

        stats_html = [
            html.H4(selected_crime, className="card-title mb-3"),
            dbc.Row([
                dbc.Col(criar_card_estatistica(
                    "Total de Ocorrências",
                    f"{filtered['Ocorrências'].sum():,.0f}",
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
        tabela_data = filtered[['Ano', 'Mês', 'Ocorrências']].to_dict('records')
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


# ========== CONFIGURAÇÃO FINAL ==========
# Verifica se o arquivo existe antes de configurar o layout
file_path = verificar_arquivo()
if file_path is None:
    app.layout = html.Div([
        html.H1("Erro de Configuração", className="text-danger"),
        html.P("O arquivo de dados não foi encontrado. Por favor, verifique se o arquivo está no diretório correto."),
        html.P("Caminhos verificados:"),
        html.Ul([
            html.Li(str(pathlib.Path(__file__).parent / "Dados Segurança Publica 2001 a 2023 consolidado.xlsx")),
            html.Li(str(pathlib.Path("Dados Segurança Publica 2001 a 2023 consolidado.xlsx"))),
            html.Li(str(pathlib.Path("pages.main") / "Dados Segurança Publica 2001 a 2023 consolidado.xlsx"))
        ]),
        html.P("Arquivos encontrados no diretório:"),
        html.Ul([html.Li(f.name) for f in pathlib.Path(__file__).parent.iterdir()])
    ])
else:
    app.layout = criar_layout()
    configurar_callbacks()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)