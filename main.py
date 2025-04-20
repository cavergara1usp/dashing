import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from datetime import datetime
import os

# ========== CONFIGURAÇÃO INICIAL ==========
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Painel de Segurança Pública (2001-2023)"


# ========== FUNÇÃO DE CARREGAMENTO DE DADOS ==========
def carregar_dados():
    try:
        meses_esperados = [
            'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
            'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'
        ]

        file_path = "Dados/Dados Segurança Publica 2001 a 2023 consolidado.xlsx"

        if not os.path.exists(file_path):
            print(f"\nERRO: Arquivo não encontrado: {file_path}")
            return None

        print(f"\nArquivo encontrado: {file_path}")

        try:
            xl = pd.ExcelFile(file_path)
            print(f"Planilhas disponíveis: {xl.sheet_names}")
        except Exception as e:
            print(f"\nERRO AO ABRIR ARQUIVO: {str(e)}")
            return None

        dfs = []
        anos_com_erro = []

        for ano in range(2001, 2024):
            sheet_name = str(ano)
            if sheet_name not in xl.sheet_names:
                print(f"Planilha {ano} não encontrada no arquivo")
                anos_com_erro.append(ano)
                continue

            try:
                # Lê a planilha inteira como texto para evitar interpretação automática
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    header=None,
                    dtype=str
                )

                # Remove linhas completamente vazias
                df = df.dropna(how='all')

                if df.empty:
                    print(f"Planilha {ano} está vazia")
                    anos_com_erro.append(ano)
                    continue

                # Encontra a linha que contém "Natureza" (cabeçalho)
                natureza_idx = df[df[0].str.contains('Natureza', na=False)].index[0]

                # Define o cabeçalho
                df.columns = df.iloc[natureza_idx]
                df = df.iloc[natureza_idx + 1:]  # Remove a linha de cabeçalho

                # Remove colunas vazias
                df = df.dropna(axis=1, how='all')

                # Renomeia colunas
                df = df.rename(columns={'Natureza': 'Natureza'})

                # Remove linhas onde Natureza está vazia
                df = df[df['Natureza'].notna()]

                # Pega apenas as colunas de meses (assumindo que são as 12 colunas após Natureza)
                meses_df = df.iloc[:, 1:13].copy()
                meses_df.columns = meses_esperados

                # Combina com a coluna Natureza
                df_final = pd.concat([df['Natureza'], meses_df], axis=1)

                # Converte valores para numérico
                for mes in meses_esperados:
                    df_final[mes] = pd.to_numeric(
                        df_final[mes].astype(str)
                        .str.replace('.', '', regex=False)
                        .str.replace(',', '.', regex=False),
                        errors='coerce'
                    )

                # Adiciona coluna de ano
                df_final['Ano'] = ano
                dfs.append(df_final)

            except Exception as e:
                print(f"Erro ao processar ano {ano}: {str(e)}")
                anos_com_erro.append(ano)
                continue

        if not dfs:
            raise ValueError("Nenhuma planilha válida foi carregada")

        # Consolida todos os DataFrames
        df_completo = pd.concat(dfs, ignore_index=True)

        # Transforma para formato longo
        df_long = df_completo.melt(
            id_vars=['Ano', 'Natureza'],
            value_vars=meses_esperados,
            var_name='Mês',
            value_name='Ocorrências'
        ).dropna()

        # Cria coluna de data completa
        mes_para_num = {mes: i + 1 for i, mes in enumerate(meses_esperados)}
        df_long['Mes_Num'] = df_long['Mês'].map(mes_para_num)
        df_long['Data'] = pd.to_datetime(
            df_long['Ano'].astype(str) + '-' +
            df_long['Mes_Num'].astype(str) + '-01'
        )

        # Remove espaços extras nos nomes dos crimes
        df_long['Natureza'] = df_long['Natureza'].str.strip()

        # Ordena por data
        df_long = df_long.sort_values('Data')

        print(f"\nDados carregados com sucesso para os anos: {sorted(set(df_long['Ano']))}")
        if anos_com_erro:
            print(f"Anos com problemas: {sorted(anos_com_erro)}")

        return df_long

    except Exception as e:
        print(f"\nERRO CRÍTICO AO CARREGAR DADOS: {str(e)}")
        return None


# ========== LAYOUT DO PAINEL ==========
def criar_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H1(
                "Análise de Criminalidade (2001-2023)",
                className="text-center my-4",
                style={'color': '#2c3e50', 'font-weight': 'bold'}
            ), width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label(
                        "Selecione o Tipo de Crime:",
                        style={'font-weight': 'bold', 'margin-bottom': '5px'}
                    ),
                    dcc.Dropdown(
                        id='crime-dropdown',
                        placeholder="Selecione um crime...",
                        style={'margin-bottom': '20px'}
                    )
                ])
            ], md=6),
            dbc.Col([
                html.Div([
                    html.Label(
                        "Selecione o Período:",
                        style={'font-weight': 'bold', 'margin-bottom': '5px'}
                    ),
                    dcc.RangeSlider(
                        id='ano-slider',
                        min=2001,
                        max=2023,
                        value=[2018, 2023],
                        marks={str(ano): {'label': str(ano), 'style': {'transform': 'rotate(45deg)'}}
                               for ano in range(2001, 2024, 2)},
                        step=1,
                        tooltip={'placement': 'bottom', 'always_visible': True},
                        allowCross=False
                    )
                ])
            ], md=6)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id='serie-temporal'), width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id='grafico-sazonal'), width=6),
            dbc.Col(dcc.Graph(id='grafico-anual'), width=6)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Estatísticas do Período", className="font-weight-bold"),
                    dbc.CardBody(id='estatisticas-resumo')
                ], color="light"),
                width=12,
                className="mb-4"
            )
        ]),
        dbc.Row([
            dbc.Col(
                html.Div(
                    f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')} | "
                    "Fonte: Dados de Segurança Pública",
                    className="text-center text-muted small"
                ),
                width=12
            )
        ])
    ], fluid=True)


# ========== CALLBACKS ==========
def configurar_callbacks():
    @app.callback(
        Output('crime-dropdown', 'options'),
        Input('ano-slider', 'value')
    )
    def atualizar_dropdown(anos_selecionados):
        df = carregar_dados()
        if df is None:
            return []
        crimes = df[df['Ano'].between(anos_selecionados[0], anos_selecionados[1])]['Natureza'].unique()
        return [{'label': crime, 'value': crime} for crime in sorted(crimes)]

    @app.callback(
        Output('serie-temporal', 'figure'),
        Output('grafico-sazonal', 'figure'),
        Output('grafico-anual', 'figure'),
        Output('estatisticas-resumo', 'children'),
        Input('crime-dropdown', 'value'),
        Input('ano-slider', 'value')
    )
    def atualizar_graficos(selected_crime, selected_years):
        if selected_crime is None:
            empty_fig = px.scatter(title="Selecione um tipo de crime")
            empty_fig.update_layout(plot_bgcolor='white')
            return empty_fig, empty_fig, empty_fig, "Selecione um tipo de crime"

        df = carregar_dados()
        if df is None:
            error_fig = px.scatter(title="Erro ao carregar dados")
            error_fig.update_layout(plot_bgcolor='white')
            return error_fig, error_fig, error_fig, "Erro ao carregar dados"

        filtered = df[
            (df['Natureza'] == selected_crime) &
            (df['Ano'].between(selected_years[0], selected_years[1]))
            ]

        if filtered.empty:
            empty_fig = px.scatter(title="Nenhum dado encontrado")
            empty_fig.update_layout(plot_bgcolor='white')
            return empty_fig, empty_fig, empty_fig, "Nenhum dado encontrado para os filtros selecionados"

        fig_serie = px.line(
            filtered,
            x='Data',
            y='Ocorrências',
            title=f'Série Temporal: {selected_crime}',
            labels={'Ocorrências': 'Número de Ocorrências', 'Data': ''},
            template='plotly_white'
        )
        fig_serie.update_traces(line=dict(width=2.5))
        fig_serie.update_layout(hovermode='x unified')

        filtered['Media_Movel'] = filtered['Ocorrências'].rolling(window=3, min_periods=1).mean()
        fig_serie.add_scatter(
            x=filtered['Data'],
            y=filtered['Media_Movel'],
            mode='lines',
            name='Média Móvel (3 meses)',
            line=dict(color='orange', width=2)
        )

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
        fig_anual.update_layout(uniformtext_minsize=8)

        stats = filtered.groupby('Ano')['Ocorrências'].sum().describe()
        ano_max = anual.loc[anual['Ocorrências'].idxmax(), 'Ano']
        ano_min = anual.loc[anual['Ocorrências'].idxmin(), 'Ano']

        stats_html = [
            html.H5(f"Análise: {selected_crime}", className="card-title"),
            html.P(f"Período: {selected_years[0]} a {selected_years[1]}", className="card-text"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Total", className="stat-title"),
                        html.P(f"{filtered['Ocorrências'].sum():,.0f}", className="stat-value")
                    ], className="stat-card")
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.H6("Média Anual", className="stat-title"),
                        html.P(f"{stats['mean']:,.0f}", className="stat-value")
                    ], className="stat-card")
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.H6("Ano com Mais Casos", className="stat-title"),
                        html.P(f"{ano_max} ({stats['max']:,.0f})", className="stat-value")
                    ], className="stat-card")
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.H6("Ano com Menos Casos", className="stat-title"),
                        html.P(f"{ano_min} ({stats['min']:,.0f})", className="stat-value")
                    ], className="stat-card")
                ], md=3)
            ])
        ]

        return fig_serie, fig_sazonal, fig_anual, stats_html


# ========== CONFIGURAÇÃO FINAL ==========
app.layout = criar_layout()
configurar_callbacks()

# ========== EXECUÇÃO ==========
server = app.server
if __name__ == '__main__':
    print("\nVerificando estrutura de dados...")
    dados = carregar_dados()
    if dados is not None:
        print("\nVisualização dos primeiros registros:")
        print(dados.head())
        print("\nInformações do DataFrame:")
        print(dados.info())
        print("\nTipos de crimes disponíveis:")
        print(dados['Natureza'].unique())

    app.run(debug=True, port=8050)