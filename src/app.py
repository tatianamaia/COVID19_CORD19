import datetime
import os
import yaml

import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

import plotly.graph_objs as go


ENV_FILE = '../env.yaml'
with open(ENV_FILE) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Initialisation des chemins vers les fichiers
ROOT_DIR = os.path.dirname(os.path.abspath(ENV_FILE))
DATA_FILE = os.path.join(ROOT_DIR,
                         params['directories']['processed'],
                         params['files']['all_data'])

#Lecture du fihcier de données
epidemie_df = (pd.read_csv(DATA_FILE, parse_dates=['Last Update'])
               .assign(day=lambda _df:_df['Last Update'].dt.date)
               .drop_duplicates(subset=['Country/Region', 'Province/State', 'day'])
               [lambda df: df['day'] <= datetime.date(2020,3,20)]
              )
# replacing Mainland china with just China 
cases = ['Confirmed', 'Deaths', 'Recovered']
# After 14/03/2020 the names of the countries are quite different 
epidemie_df['Country/Region'] = epidemie_df['Country/Region'].replace('Mainland China', 'China')
# filling missing values 
epidemie_df[['Province/State']] = epidemie_df[['Province/State']].fillna('')
epidemie_df[cases] = epidemie_df[cases].fillna(0)

countries=[{'label':c, 'value': c} for c in epidemie_df['Country/Region'].unique()]

app = dash.Dash('C0VID-19 Explorer')

app.layout = html.Div([
    html.H1(['C0VID-19 Explorer'], style={'textAlign': 'center', 'color': 'navy', 'font-weight': 'bold'}),
    dcc.Tabs([
        dcc.Tab(label='Time', children=[
            dcc.Markdown("""
                Select a country:
              
            """,style={'textAlign': 'left', 'color': 'navy', 'font-weight': 'bold'} ),
            html.Div([
                dcc.Dropdown(
                    id='country',
                    options=countries,
                    placeholder="Select a country...",
                )
            ]),
            html.Div([
                dcc.Markdown("""You can select a second country:""", 
                             style={'textAlign': 'left', 'color': 'navy', 'font-weight': 'bold'} ),
                dcc.Dropdown(
                    id='country2',
                    options=countries,
                    placeholder="Select a country...",
                )

            ]),
            html.Div([dcc.Markdown("""Cases: """, 
                             style={'textAlign': 'left', 'color': 'navy', 'font-weight': 'bold'} ),
                dcc.RadioItems(
                    id='variable',
                    options=[
                        {'label':'Confirmed', 'value': 'Confirmed'},
                        {'label':'Deaths', 'value': 'Deaths'},
                        {'label':'Recovered', 'value': 'Recovered'}
                    ],
                    value='Confirmed',
                    labelStyle={'display': 'inline-block'}
                )
            ]),
            html.Div([
                dcc.Graph(id='graph1')
            ])
        ]),
        dcc.Tab(label='Map', children=[
            #html.H6(['COVID-19 in numbers:']),
            dcc.Markdown("""
                **COVID-19**
               This is a graph that shows the evolution of the COVID-19 around the world 
               
               ** Cases:**
            """, style={'textAlign': 'left', 'color': 'navy', 'font-weight': 'bold'} ),
            dcc.Dropdown(id="value-selected", value='Confirmed',
                         options=[{'label': "Deaths ", 'value': 'Deaths'},
                                  {'label': "Confirmed", 'value': 'Confirmed'},
                                  {'label': "Recovered", 'value': 'Recovered'}],
                         placeholder="Select a country...",
                         style={"display": "inline-block", "margin-left": "auto", "margin-right": "auto",
                                "width": "70%"}, className="six columns"),
            
            dcc.Graph(id='map1'),
            dcc.Slider(
                id='map_day',
                min=0,
                max=(epidemie_df['day'].max() - epidemie_df['day'].min()).days,
                value=0,
                marks={i:str(i) for i, date in enumerate(epidemie_df['day'].unique())}
            )     
        ]),
        dcc.Tab(label='SIR Model', children=[
            dcc.Markdown("""
                **SIR model**
               S(Susceptible)I(Infectious)R(Recovered) is a model describing the dynamics of infectious disease. The model divides the population into compartments. Each compartment is expected to have the same characteristics. SIR represents the three compartments segmented by the model.
               
               **Select a country:**
            """, style={'textAlign': 'left', 'color': 'navy'}),
            html.Div([
                dcc.Dropdown(
                    id='Country',
                    value='Portugal',
                    options=countries),
            ]),
            dcc.Markdown("""Select:""", style={'textAlign': 'left', 'color': 'navy'}),
            dcc.Dropdown(id='cases',
                options=[
                    {'label': 'Confirmed', 'value': 'Confirmed'},
                    {'label': 'Deaths', 'value': 'Deaths'},
                    {'label': 'Recovered', 'value': 'Recovered'}],
                         value=['Confirmed','Deaths','Recovered'],
                multi=True),
    
             dcc.Markdown("""
             
             **Select your paramaters:**
             
            """, style={'textAlign': 'left', 'color': 'navy'}),
            html.Label( style={'textAlign': 'left', 'color': 'navy', "width": "20%"}),
            html.Div([
                 dcc.Markdown(""" Beta:         
            """, style={'textAlign': 'left', 'color': 'navy'}),
                dcc.Input(
                    id='input-beta',
                    type ='number',
                    placeholder='Input Beta',
                    min  =-50, 
                    max  =100,
                    step =0.01,
                    value=0.45
                ) 
               ]),
            html.Div([
                dcc.Markdown(""" Gamma:         
            """, style={'textAlign': 'left', 'color': 'navy'}),
                dcc.Input(
                    id='input-gamma',
                    type ='number',
                    placeholder='Input Gamma',
                    min  =-50, 
                    max  =100,
                    step =0.01,
                    value=0.55
                )
            ]),
            html.Div([
                dcc.Markdown(""" Population:         
            """, style={'textAlign': 'left', 'color': 'navy'}),
                dcc.Input(
                    id='input-pop',placeholder='Population',
                    type ='number',
                    min  =1000, 
                    max  =1000000000000000,
                    step =1000,
                    value=1000,
                )
            ]),
            html.Div([
                dcc.RadioItems(id='variable2',
                              options=[
                              {'label':'Optimize','value':'optimize'}],
                              value='Confirmed',
                              labelStyle={'display':'inline-block','color': 'navy', "width": "20%"})
                ]),
            html.Div([
                dcc.Graph(id='graph2')
            ]),
            
        ])
    ]),
])


@app.callback(
    Output('graph1', 'figure'),
    [
        Input('country','value'),
        Input('country2','value'),
        Input('variable','value'),
    ]
)
def update_graph(country, country2, variable):
    print(country)
    
    if country is None:
        graph_df = epidemie_df.groupby('day').agg({variable:'sum'}).reset_index()
    else:
            graph_df=(epidemie_df[epidemie_df['Country/Region'] == country]
              .groupby(['Country/Region', 'day'])
              .agg({variable:'sum'})
              .reset_index()
             )
    if country2 is not None:
        graph2_df=(epidemie_df[epidemie_df['Country/Region'] == country2]
              .groupby(['Country/Region', 'day'])
              .agg({variable:'sum'})
              .reset_index()
             )              
    return {
        'data':[
            dict(
                x=graph_df['day'],
                y=graph_df[variable],
                type='line',
                name=country if country is not None else 'Total'
            )
        ] + ([
            dict(
                x=graph2_df['day'],
                y=graph2_df[variable],
                type='line',
                name=country2
            )
        ] if country2 is not None else [])
    }

@app.callback(
    Output('map1', 'figure'),
    [
        Input('map_day','value'),
        Input("value-selected", "value")
    ]
)
def update_map(map_day,selected):
    day= epidemie_df['day'].sort_values(ascending=False).unique()[map_day]
    map_df = (epidemie_df[epidemie_df['day'] == day]
              .groupby(['Country/Region'])
              .agg({selected:'sum', 'Latitude': 'mean', 'Longitude': 'mean'})
              .reset_index()
             )

    return {
        'data':[
            dict(
                type='scattergeo',
                lon=map_df['Longitude'],
                lat=map_df['Latitude'],
                text=map_df.apply(lambda r: r['Country/Region'] + '(' + str(r[selected]) + ')', axis=1),
                mode='markers',
                marker=dict(
                    size=np.maximum(map_df[selected]/ 1_000, 10)
                )
            )
        ],
        'layout': dict(
            title=str(day),
            geo=dict(showland=True)
        )
       
    }

@app.callback(
    Output('graph2', 'figure'),
    [
        Input('input-beta', 'value'),
        Input('input-gamma','value'),
        Input('input-pop','value'),
        Input('Country','value')
        #Input('variable2','value')
        
    ]
)
def update_model(beta, gamma, population, Country):
    print(Country)
    
    country=Country
    country_df = (epidemie_df[epidemie_df['Country/Region'] == country]
                   .groupby(['Country/Region', 'day'])
                   .agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})
                   .reset_index())
    country_df['Infected'] = country_df['Confirmed'].diff()
        
    steps = len(country_df['Infected'])

    
    def SIR(t, y):
        S = y[0]; I = y[1]; R = y[2]
        return([-beta*S*I, beta*S*I-gamma*I, gamma*I]) 
    
    solution = solve_ivp(SIR, [0, steps], [population, 1, 0], t_eval=np.arange(0, steps, 1))
    
    #def sumsq_error(parameters):
        #beta, gamma = parameters
        #def SIR(t,y):
            #S=y[0]
            #I=y[1]
            #R=y[2]
        #return([-beta*S*I, beta*S*I-gamma*I, gamma*I])
    
    #solution = solve_ivp(SIR,[0,nb_steps-1],[total_population,1,0],t_eval=np.arange(0,nb_steps,1))
        
    #return(sum((solution.y[1]-infected_population)**2))
    
    #msol = minimize(sumsq_error,[0.001,0.1],method='Nelder-Mead')

    #if variable2 == 'optimize':
        #gamma,beta == msol.x

    return {
        'data': [
            dict(
                x=solution.t,
                y=solution.y[0],
                type='line',
                name=country+': Susceptible')
        ] + ([
            dict(
                x=solution.t,
                y=solution.y[1],
                type='line',
                name=country+': Infected')
        ]) + ([
            dict(
                x=solution.t,
                y=solution.y[2],
                type='line',
                name=country+': Recovered')
        ]) + ([
            dict(
                x=solution.t,
                y=country_df['Infected'],
                type='line',
                name=country+': Original Data(Infected)')
        ])
    }


if __name__ == '__main__':
    app.run_server(debug=True)