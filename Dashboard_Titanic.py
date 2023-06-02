
import dash
from dash import Dash, html, dcc, Input, Output 

#FOR CALLBACK 

from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go 
import dash_table
import pandas as pd 
import numpy as np
 
import random as rnd
 

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#from plotly import graph_objs as go
from datetime import datetime as dt
import json 

from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme, Sign

################### Data Part #####################

train_df = pd.read_csv('../data/titanic/train.csv')
test_df = pd.read_csv('../data/titanic//test.csv')
combine = [train_df, test_df]


### plots
# survival plot
survivalPlot = px.bar(train_df, x='Sex', y='Survived')
#survivalPlot.show()

#histogram_plot for age vs. survival
age_hist_plot = px.histogram(train_df, x='Age', color='Survived')

# pclass bubble plot
pclass_survival_stat = train_df[['Pclass', 'Survived','Sex']].groupby(['Pclass','Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False) 
pclass_bubble_plot=px.scatter(pclass_survival_stat, x='Pclass', y='Survived', size="Survived", color="Sex",   hover_name="Pclass", log_x=True, size_max=60)
pclass_bubble_plot.update_xaxes(type='category')

#################### Dash Visualization Part ###################

#https://towardsdatascience.com/3-easy-ways-to-make-your-dash-application-look-better-3e4cfefaf772




#app.server.run()
#app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
 
app = dash.Dash('Titanic Analysis', external_stylesheets=[dbc.themes.LUX])  #external_stylesheets=['https://example.com/style1.css', 'https://example.com/style2.css'])
server=app.server
load_figure_template('LUX')

 

#app = dash.Dash('Titanic Analysis Demo',external_stylesheets=['https://example.com/style1.css', 'https://example.com/style2.css'])

    
external_css = [ "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
        "//fonts.googleapis.com/css?family=Raleway:400,300,600",
        "https://codepen.io/plotly/pen/KmyPZr.css",
        "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"]

for css in external_css: 
    app.css.append_css({ "external_url": css })
    
external_js = [ "https://code.jquery.com/jquery-3.2.1.min.js",
        "https://codepen.io/plotly/pen/KmyPZr.js" ]
    
for js in external_js: 
    app.scripts.append_script({ "external_url": js })
 
app.css.config.serve_locally = False
# Describe the layout, or the UI, of the app
app.layout = html.Div([   
   
    #style={'backgroundColor': '#f2f2f2', 'height': '100vh','paddingLeft': '20px'},
    #style={'paddingLeft': '20px'},  # Adjust the padding value as needed
#    children=[
    
        html.Div([ # page 1

            html.A([ 'Print PDF' ], className="button no-print",  style=dict(position="absolute", top=-40, right=0)),     

           
            html.Div([ # subpage 1

                
                # Row 1 (Header)

                html.Div([
                    html.Div([      
                        html.H3('Titanic Data Analysis', style={'textAlign': 'center' ,'color':'darkmagenta' }),
                        html.H4('This is a demo to show the analysis for Titanic Data',  style={'textAlign': 'center' ,'color':'darkmagenta' ,'paddingleft':'20px'}), #color='#7F90AC')),
                    ], className = "nine columns padded" ),
               html.Div([            
                    html.H1([html.Span('LinkedKey Python Course', style=dict(opacity=0.2))]), #, html.Span('17')]),
           
                ], className = "three columns gs-header gs-accent-header padded", style=dict(float='right')),


                ], className = "row gs-header gs-text-header" ),
 
                html.Br([]),   
                
                html.Div([
                      
                    html.H5("The sinking of the Titanic is one of the most infamous shipwrecks in history."),
                    html.P("On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew."),
                    html.P("While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc)"),
                    html.Hr(style={'borderWidth': "0.1vh", "width": "100%", "borderColor": "#53917E", "borderStyle":"solid"}), #,width={'size':10, 'offset':1}),
                ]),
                
                 # Row 2

                html.Div([     

                    html.Div([
                       # style={'paddingLeft':'20px'},
                        html.H5('Titanic Basic Statistics', className = "gs-header gs-text-header padded",style={'paddingLeft': '20px'}),

                        html.Strong('Overall Stats',style={'paddingLeft': '20px'}),
                        html.Br([]),
                        html.Ul([
                            html.Li('Total Training Sample size : 891. ', className = 'blue-text',style={'paddingLeft': '20px'}),
                            html.Li('Survival Rate: 32%. ', className = 'blue-text',style={'paddingLeft': '20px'}),
                        ]),

                        html.Strong('Details',style={'paddingLeft': '20px'}),
                        html.Ul([
                            html.Li('Most passengers (>75%) did not travel with parents or children.  ', className = 'blue-text',style={'paddingLeft': '20px'}),
                            html.Li('Nearly 30% of the passengers had sibliings and/ or spouse aboard. ', className = 'blue-text',style={'paddingLeft': '20px'}),
                            html.Li('Fares varied significantely with few passengers paying as high as $512.  ', className = 'blue-text',style={'paddingLeft': '20px'}),
                            html.Li('Few elderly passengers within age range 65-80 ', className = 'blue-text',style={'paddingLeft': '20px'}),
                        ]),

                        html.Strong('Male v.s Female:',style={'paddingLeft': '20px'}),
                        html.Ul([
                            html.Li('Passengers:: 65% vs. 35%',style={'paddingLeft': '20px'}),
                            
                            html.Li('Survival Rate:: 18% vs. 74%',style={'paddingLeft': '20px'}),
                        ]), 

                    ], className = "four columns" ),

                    html.Div([
                        html.H5(["Survival by Gender"],  
                                className = "gs-header gs-table-header padded"),
                        dcc.Graph(id ='survival by gender', figure=survivalPlot),
                        html.P("This is number of passengers survived distributed by Gender. Over 70% of female passengers survived, while under 20% of male are survived  ."),
                    ], className = "eight columns" ),

                ], className = "row "),  

                # Row 2.5

                html.Div([     

                    html.Div([
                        html.H5("Interesitng Facts" ,  className = "gs-header gs-text-header padded",style={'paddingLeft': '20px'}),
                        html.Br([]),
                        html.Strong("Survival rate varies by Age",style={'paddingLeft': '20px'}),
                        
                        html.Ul([
                            html.Li("Infants (Age <=4) had high survival rate.",style={'paddingLeft': '20px'}),
                            html.Li("Oldest passengers (Age = 80) survived.",style={'paddingLeft': '20px'}),
                            html.Li("Large number of 15-25 year olds did not survive.",style={'paddingLeft': '20px'}),
                            html.Li("Most passengers are in 15-35 age range.",style={'paddingLeft': '20px'})
                        ]), 
                    ], className = "four columns" ),

                    html.Div([
                       html.H5(["Survival by Age"],  
                                className = "gs-header gs-table-header padded"),
                        dcc.Graph(id ='survival by age', figure=age_hist_plot),
                        html.P(" ."),

                    ], className = "eight columns" ),

                ], className = "row "),  #end row 2.5
                
                
                # Row 3

                html.Div([     

                    html.Div([
                        #html.H5("Interesitng Fact" ,  className = "gs-header gs-text-header padded",style={'paddingLeft': '20px'}),
                        html.Strong("Passengers in different class has different survival rate",style={'paddingLeft': '20px'}),
                        html.Ul([
                            html.Li("Pclass=3 had most passengers, however most did not survive.",style={'paddingLeft': '20px'}),
                            html.Li("Most passengers in Pclass=1 survived",style={'paddingLeft': '20px'}),
                            html.Li("Pclass varies in terms of Age distribution of passengers",style={'paddingLeft': '20px'}), 
                        ]), 
                    ], className = "four columns" ),

                    html.Div([
                       html.H5(["Survival by Pclass"], className = "gs-header gs-table-header padded"),
                                
                        dcc.Graph(id ='survival by Pclass', figure=pclass_bubble_plot),
                       

                    ], className = "eight columns" ),

                ], className = "row "),  #end row 2.5



            ], className = "subpage" ),

        ]),
   # ]
])



 
app.server.run() 
