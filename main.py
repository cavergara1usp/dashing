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
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import shapiro, ttest_ind, zscore
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ========== CONFIGURAÇÃO INICIAL ==========
app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])
app.title = "Painel de Segurança Pública (2001-2023) - Análise Estatística Avançada"
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
    """Calcula a regressão linear e média móvel para um crime específico com intervalos de confiança."""
    dados = df[df['Natureza'] == crime_selecionado]
    dados_anuais = dados.groupby('Ano')['Ocorrências'].sum().reset_index()

    x = dados_anuais['Ano'].values
    y = dados_anuais['Ocorrências'].values

    if len(x) < 2:
        return None

    # Regressão linear com intervalo de confiança
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    linha_regressao = intercept + slope * x

    # Intervalo de confiança (95%)
    n = len(x)
    t = stats.t.ppf(0.975, df=n - 2)  # Valor t para 95% de confiança
    ci = t * std_err * np.sqrt(1 / n + (x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))

    # Teste de normalidade nos resíduos
    residuos = y - linha_regressao
    _, p_normalidade = shapiro(residuos)

    # Média móvel
    media_movel = pd.Series(y).rolling(window=3, center=True, min_periods=1).mean().values

    return {
        'Anos': x,
        'Ocorrências': y,
        'Regressão': linha_regressao,
        'CI_upper': linha_regressao + ci,
        'CI_lower': linha_regressao - ci,
        'Média Móvel': media_movel,
        'Coeficiente': slope,
        'Intercepto': intercept,
        'R²': r_value ** 2,
        'p_value': p_value,
        'p_normalidade': p_normalidade,
        'std_err': std_err
    }


def decompor_serie_temporal(df, crime_selecionado):
    """Realiza decomposição sazonal da série temporal."""
    try:
        dados = df[df['Natureza'] == crime_selecionado]
        serie = dados.set_index('Data')['Ocorrências'].sort_index()

        # Preenche possíveis datas faltantes
        date_range = pd.date_range(start=serie.index.min(), end=serie.index.max(), freq='MS')
        serie = serie.reindex(date_range).fillna(method='ffill').fillna(method='bfill')

        decomposicao = seasonal_decompose(serie, model='additive', period=12)
        return decomposicao
    except Exception as e:
        logger.error(f"Erro na decomposição: {str(e)}")
        return None


def detectar_outliers(df, crime_selecionado):
    """Identifica outliers usando o método Z-Score."""
    dados = df[df['Natureza'] == crime_selecionado]

    # Calcula z-scores por mês para considerar sazonalidade
    outliers = []
    for mes in dados['Mês'].unique():
        subset = dados[dados['Mês'] == mes]
        z_scores = zscore(subset['Ocorrências'])
        subset_outliers = subset[(abs(z_scores) > 3)].copy()
        subset_outliers['Z-Score'] = z_scores[abs(z_scores) > 3]
        outliers.append(subset_outliers)

    if outliers:
        return pd.concat(outliers)
    return pd.DataFrame()


def comparar_periodos(df, crime_selecionado, periodo1, periodo2):
    """Compara estatisticamente dois períodos usando teste t."""
    dados = df[df['Natureza'] == crime_selecionado]

    p1 = dados[dados['Ano'].between(periodo1[0], periodo1[1])]['Ocorrências']
    p2 = dados[dados['Ano'].between(periodo2[0], periodo2[1])]['Ocorrências']

    if len(p1) < 2 or len(p2) < 2:
        return None

    t_stat, p_val = ttest_ind(p1, p2, equal_var=False)

    return {
        'periodo1': {'média': p1.mean(), 'desvio': p1.std(), 'n': len(p1)},
        'periodo2': {'média': p2.mean(), 'desvio': p2.std(), 'n': len(p2)},
        't_stat': t_stat,
        'p_val': p_val,
        'diferenca': p2.mean() - p1.mean(),
        'diferenca_percentual': (p2.mean() - p1.mean()) / p1.mean() * 100
    }


def prever_arima(df, crime_selecionado, passos=12):
    """Realiza previsão usando modelo ARIMA."""
    try:
        dados = df[df['Natureza'] == crime_selecionado]
        serie = dados.set_index('Data')['Ocorrências'].sort_index()

        # Preenche possíveis datas faltantes
        date_range = pd.date_range(start=serie.index.min(), end=serie.index.max(), freq='MS')
        serie = serie.reindex(date_range).fillna(method='ffill').fillna(method='bfill')

        # Modelo ARIMA simples (ordens podem ser otimizadas)
        model = ARIMA(serie, order=(1, 1, 1))
        model_fit = model.fit()

        # Previsão
        forecast = model_fit.get_forecast(steps=passos)
        pred = forecast.predicted_mean
        ci = forecast.conf_int()

        return {
            'historico': serie,
            'previsao': pred,
            'ci_upper': ci.iloc[:, 1],
            'ci_lower': ci.iloc[:, 0],
            'modelo': model_fit
        }
    except Exception as e:
        logger.error(f"Erro na previsão ARIMA: {str(e)}")
        return None


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
                    dbc.Col(
                        html.H1("Painel de Criminalidade - Análise Estatística Avançada", className="text-center my-4"),
                        width=12),
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

                # Gráfico de Regressão com Intervalo de Confiança
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader("Análise de Tendência com Intervalo de Confiança", className="fw-bold"),
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
                            dbc.CardHeader("Estatísticas Descritivas", className="fw-bold"),
                            dbc.CardBody(id='estatisticas-resumo')
                        ]),
                        width=12
                    ),
                    className="mb-4"
                ),

                # Nova Seção: Análise Estatística Avançada
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader("Análise Estatística Avançada", className="fw-bold"),
                            dbc.CardBody([
                                dbc.Tabs([
                                    # Tab 1: Decomposição Sazonal
                                    dbc.Tab(
                                        dcc.Graph(id='decomposicao-sazonal'),
                                        label="Decomposição Sazonal"
                                    ),

                                    # Tab 2: Detecção de Outliers
                                    dbc.Tab(
                                        html.Div([
                                            dash_table.DataTable(
                                                id='tabela-outliers',
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
                                            html.P("Outliers identificados usando Z-Score > 3",
                                                   className="text-muted mt-2")
                                        ]),
                                        label="Outliers"
                                    ),

                                    # Tab 3: Comparação de Períodos
                                    dbc.Tab(
                                        html.Div([
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Label("Período 1:", className="fw-bold"),
                                                    dcc.RangeSlider(
                                                        id='periodo1-slider',
                                                        min=2001,
                                                        max=2023,
                                                        value=[2001, 2005],
                                                        marks={str(ano): {'label': str(ano)} for ano in
                                                               range(2001, 2024, 2)},
                                                        step=1
                                                    )
                                                ], md=6),

                                                dbc.Col([
                                                    html.Label("Período 2:", className="fw-bold"),
                                                    dcc.RangeSlider(
                                                        id='periodo2-slider',
                                                        min=2001,
                                                        max=2023,
                                                        value=[2018, 2023],
                                                        marks={str(ano): {'label': str(ano)} for ano in
                                                               range(2001, 2024, 2)},
                                                        step=1
                                                    )
                                                ], md=6)
                                            ]),
                                            html.Div(id='comparacao-periodos', className="mt-4")
                                        ]),
                                        label="Comparar Períodos"
                                    ),

                                    # Tab 4: Previsão ARIMA
                                    dbc.Tab(
                                        dcc.Graph(id='previsao-arima'),
                                        label="Previsão"
                                    )
                                ])
                            ])
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


# ========== CONFIGURAÇÃO DOS CALLBACKS ==========
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
            return empty, empty, empty, "Selecione um tipo de crime", [], columns

        df = pd.DataFrame.from_records(dados)
        filtered = df[(df['Natureza'] == selected_crime) &
                      (df['Ano'].between(selected_years[0], selected_years[1]))]

        if filtered.empty:
            empty = px.scatter(title="Nenhum dado encontrado")
            empty.update_layout(template='plotly_white')

            columns = [{'name': col, 'id': col} for col in ['Ano', 'Mês', 'Ocorrências']]
            return empty, empty, empty, "Nenhum dado encontrado", [], columns

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

        # 4. Estatísticas Resumidas
        stats = filtered.groupby('Ano')['Ocorrências'].sum().describe()
        regressao = calcular_regressao_media(filtered, selected_crime)
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

        # 5. Tabela de dados
        tabela_data = filtered[['Ano', 'Mês', 'Ocorrências']].to_dict('records')
        tabela_columns = [{'name': col, 'id': col} for col in ['Ano', 'Mês', 'Ocorrências']]

        return fig_serie, fig_sazonal, fig_anual, stats_html, tabela_data, tabela_columns

    @app.callback(
        [Output('grafico-regressao', 'figure'),
         Output('regressao-stats', 'children')],
        [Input('crime-dropdown', 'value'),
         Input('ano-slider', 'value')],
        State('dados-store', 'data')
    )
    def atualizar_regressao(selected_crime, selected_years, dados):
        if not selected_crime or not dados:
            empty = px.scatter(title="Selecione um tipo de crime")
            return empty, "Selecione um tipo de crime"

        df = pd.DataFrame.from_records(dados)
        filtered = df[(df['Natureza'] == selected_crime) &
                      (df['Ano'].between(selected_years[0], selected_years[1]))]

        regressao = calcular_regressao_media(filtered, selected_crime)

        if not regressao:
            empty = px.scatter(title="Dados insuficientes para análise")
            return empty, "Dados insuficientes"

        # Gráfico com intervalo de confiança
        fig = go.Figure()

        # Dados observados
        fig.add_trace(go.Scatter(
            x=regressao['Anos'],
            y=regressao['Ocorrências'],
            mode='markers',
            name='Dados Observados',
            marker=dict(color='blue', size=8)
        ))

        # Linha de regressão
        fig.add_trace(go.Scatter(
            x=regressao['Anos'],
            y=regressao['Regressão'],
            mode='lines',
            name='Tendência Linear',
            line=dict(color='red', width=3)
        ))

        # Intervalo de confiança
        fig.add_trace(go.Scatter(
            x=np.concatenate([regressao['Anos'], regressao['Anos'][::-1]]),
            y=np.concatenate([regressao['CI_upper'], regressao['CI_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='IC 95%'
        ))

        # Média móvel
        fig.add_trace(go.Scatter(
            x=regressao['Anos'],
            y=regressao['Média Móvel'],
            mode='lines',
            name='Média Móvel (3 anos)',
            line=dict(color='green', width=2, dash='dot')
        ))

        fig.update_layout(
            title=f'Tendência Anual com Intervalo de Confiança - {selected_crime}',
            xaxis_title='Ano',
            yaxis_title='Ocorrências',
            template='plotly_white',
            hovermode='x unified'
        )

        # Estatísticas formatadas
        stats_html = [
            html.H5("Estatísticas da Regressão", className="mt-3"),
            dbc.Row([
                dbc.Col(criar_card_estatistica(
                    "Coeficiente Angular",
                    f"{regressao['Coeficiente']:.2f}",
                    "primary"
                ), md=3),

                dbc.Col(criar_card_estatistica(
                    "Valor-p",
                    f"{regressao['p_value']:.4f}",
                    "danger" if regressao['p_value'] > 0.05 else "success"
                ), md=3),

                dbc.Col(criar_card_estatistica(
                    "R²",
                    f"{regressao['R²']:.3f}",
                    "warning"
                ), md=3),

                dbc.Col(criar_card_estatistica(
                    "Normalidade (Shapiro-Wilk)",
                    f"p={regressao['p_normalidade']:.4f}",
                    "danger" if regressao['p_normalidade'] < 0.05 else "success"
                ), md=3)
            ]),

            html.P("Nota: p-valor < 0.05 indica significância estatística", className="text-muted mt-2"),
            html.P(f"Erro padrão: {regressao['std_err']:.2f}", className="text-muted")
        ]

        return fig, stats_html

    @app.callback(
        Output('decomposicao-sazonal', 'figure'),
        [Input('crime-dropdown', 'value'),
         Input('ano-slider', 'value')],
        State('dados-store', 'data')
    )
    def atualizar_decomposicao(selected_crime, selected_years, dados):
        if not selected_crime or not dados:
            return px.scatter(title="Selecione um tipo de crime")

        df = pd.DataFrame.from_records(dados)
        filtered = df[(df['Natureza'] == selected_crime) &
                      (df['Ano'].between(selected_years[0], selected_years[1]))]

        decomposicao = decompor_serie_temporal(filtered, selected_crime)

        if not decomposicao:
            return px.scatter(title="Dados insuficientes para decomposição")

        # Criar figura com subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=('Observado', 'Tendência', 'Sazonalidade', 'Resíduos')
        )

        # Componentes da decomposição
        fig.add_trace(
            go.Scatter(x=decomposicao.observed.index, y=decomposicao.observed, name='Observado'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=decomposicao.trend.index, y=decomposicao.trend, name='Tendência'),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=decomposicao.seasonal.index, y=decomposicao.seasonal, name='Sazonalidade'),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(x=decomposicao.resid.index, y=decomposicao.resid, name='Resíduos'),
            row=4, col=1
        )

        fig.update_layout(
            height=800,
            title_text=f"Decomposição Sazonal - {selected_crime}",
            template='plotly_white',
            showlegend=False
        )

        return fig

    @app.callback(
        [Output('tabela-outliers', 'data'),
         Output('tabela-outliers', 'columns')],
        [Input('crime-dropdown', 'value'),
         Input('ano-slider', 'value')],
        State('dados-store', 'data')
    )
    def atualizar_outliers(selected_crime, selected_years, dados):
        if not selected_crime or not dados:
            return [], [{'name': 'Ano', 'id': 'Ano'}, {'name': 'Mês', 'id': 'Mês'}]

        df = pd.DataFrame.from_records(dados)
        filtered = df[(df['Natureza'] == selected_crime) &
                      (df['Ano'].between(selected_years[0], selected_years[1]))]

        outliers = detectar_outliers(filtered, selected_crime)

        if outliers.empty:
            return [], [{'name': 'Ano', 'id': 'Ano'}, {'name': 'Mês', 'id': 'Mês'}]

        outliers_data = outliers[['Ano', 'Mês', 'Ocorrências', 'Z-Score']].to_dict('records')
        columns = [{'name': col, 'id': col} for col in ['Ano', 'Mês', 'Ocorrências', 'Z-Score']]

        return outliers_data, columns

    @app.callback(
        Output('comparacao-periodos', 'children'),
        [Input('crime-dropdown', 'value'),
         Input('periodo1-slider', 'value'),
         Input('periodo2-slider', 'value')],
        State('dados-store', 'data')
    )
    def atualizar_comparacao(selected_crime, periodo1, periodo2, dados):
        if not selected_crime or not dados:
            return "Selecione um tipo de crime"

        df = pd.DataFrame.from_records(dados)
        comparacao = comparar_periodos(df, selected_crime, periodo1, periodo2)

        if not comparacao:
            return "Dados insuficientes para comparação"

        p1 = comparacao['periodo1']
        p2 = comparacao['periodo2']

        # Determinar se a diferença é significativa
        significativo = comparacao['p_val'] < 0.05
        cor_significancia = "success" if significativo else "danger"
        texto_significancia = "SIM" if significativo else "NÃO"

        return [
            html.H5(f"Comparação entre {periodo1[0]}-{periodo1[1]} e {periodo2[0]}-{periodo2[1]}", className="mb-3"),

            dbc.Row([
                dbc.Col(criar_card_estatistica(
                    f"Média {periodo1[0]}-{periodo1[1]}",
                    f"{p1['média']:,.1f} ± {p1['desvio']:,.1f}",
                    "info"
                ), md=4),

                dbc.Col(criar_card_estatistica(
                    f"Média {periodo2[0]}-{periodo2[1]}",
                    f"{p2['média']:,.1f} ± {p2['desvio']:,.1f}",
                    "info"
                ), md=4),

                dbc.Col(criar_card_estatistica(
                    "Diferença",
                    f"{comparacao['diferenca']:,.1f} ({comparacao['diferenca_percentual']:+.1f}%)",
                    "primary" if comparacao['diferenca'] > 0 else "danger"
                ), md=4)
            ]),

            dbc.Row([
                dbc.Col(criar_card_estatistica(
                    "Teste t (Welch)",
                    f"t = {comparacao['t_stat']:.2f}",
                    "secondary"
                ), md=4),

                dbc.Col(criar_card_estatistica(
                    "Valor-p",
                    f"{comparacao['p_val']:.4f}",
                    cor_significancia
                ), md=4),

                dbc.Col(criar_card_estatistica(
                    "Diferença Significativa?",
                    texto_significancia,
                    cor_significancia
                ), md=4)
            ]),

            html.P("Nota: p-valor < 0.05 indica diferença estatisticamente significativa", className="text-muted mt-3")
        ]

    @app.callback(
        Output('previsao-arima', 'figure'),
        [Input('crime-dropdown', 'value'),
         Input('ano-slider', 'value')],
        State('dados-store', 'data')
    )
    def atualizar_previsao(selected_crime, selected_years, dados):
        if not selected_crime or not dados:
            return px.scatter(title="Selecione um tipo de crime")

        df = pd.DataFrame.from_records(dados)
        filtered = df[(df['Natureza'] == selected_crime) &
                      (df['Ano'].between(selected_years[0], selected_years[1]))]

        previsao = prever_arima(filtered, selected_crime)

        if not previsao:
            return px.scatter(title="Dados insuficientes para previsão")

        # Criar figura
        fig = go.Figure()

        # Histórico
        fig.add_trace(go.Scatter(
            x=previsao['historico'].index,
            y=previsao['historico'],
            mode='lines',
            name='Histórico',
            line=dict(color='blue', width=2)
        ))

        # Previsão
        fig.add_trace(go.Scatter(
            x=previsao['previsao'].index,
            y=previsao['previsao'],
            mode='lines',
            name='Previsão',
            line=dict(color='red', width=2, dash='dot')
        ))

        # Intervalo de confiança
        fig.add_trace(go.Scatter(
            x=np.concatenate([previsao['previsao'].index, previsao['previsao'].index[::-1]]),
            y=np.concatenate([previsao['ci_upper'], previsao['ci_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='IC 95%'
        ))

        fig.update_layout(
            title=f'Previsão ARIMA para {selected_crime}',
            xaxis_title='Data',
            yaxis_title='Ocorrências',
            template='plotly_white',
            hovermode='x unified'
        )

        return fig

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
app.layout = criar_layout()
configurar_callbacks()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)