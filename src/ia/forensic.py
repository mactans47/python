import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

# Cargar los datos del tráfico de red desde un archivo CSV
df = pd.read_csv('network_traffic.csv')

# Convertir la columna de tiempo en formato datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Layout de la aplicación
app.layout = html.Div(children=[
    html.H1(children='Dashboard de Análisis Forense de Tráfico de Red'),

    html.Div(children='''
        Este dashboard interactivo permite visualizar patrones y posibles anomalías en el tráfico de red.
    '''),

    # Dropdown para seleccionar el protocolo (TCP, UDP, etc.)
    html.Label('Selecciona el protocolo'),
    dcc.Dropdown(
        id='protocol-dropdown',
        options=[{'label': proto, 'value': proto} for proto in df['protocol'].unique()],
        value='TCP',
        clearable=False
    ),

    # Gráfico de dispersión del tráfico de red
    dcc.Graph(id='scatter-plot'),

    # Histograma del tamaño de los paquetes
    dcc.Graph(id='histogram-plot')
])

# Callbacks para actualizar los gráficos de forma dinámica según el protocolo seleccionado
@app.callback(
    [dash.dependencies.Output('scatter-plot', 'figure'),
     dash.dependencies.Output('histogram-plot', 'figure')],
    [dash.dependencies.Input('protocol-dropdown', 'value')]
)
def update_graphs(selected_protocol):
    # Filtrar el DataFrame según el protocolo seleccionado
    filtered_df = df[df['protocol'] == selected_protocol]

    # Crear gráfico de dispersión (paquetes en el tiempo)
    scatter_fig = px.scatter(filtered_df, x='timestamp', y='packet_size',
                             color='source_ip', title=f'Tráfico de Red para {selected_protocol}',
                             labels={'packet_size': 'Tamaño del Paquete (bytes)', 'timestamp': 'Tiempo'},
                             hover_data=['source_ip', 'destination_ip'])

    # Crear histograma del tamaño de los paquetes
    hist_fig = px.histogram(filtered_df, x='packet_size', nbins=50,
                            title=f'Histograma del Tamaño de Paquetes para {selected_protocol}',
                            labels={'packet_size': 'Tamaño del Paquete (bytes)'})

    return scatter_fig, hist_fig


# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)