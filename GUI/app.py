# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import time
import base64
import os
import glob
import sys
sys.path.insert(0, os.getcwd())
from build_vocab import Vocabulary
from sample import get_caption

app = dash.Dash(__name__)
if not os.path.exists(f'{os.getcwd()}/GUI/assets/static/'):
    os.mkdir(f'{os.getcwd()}/GUI/assets/static/')

def save_file(name, content):
    if len(os.listdir(f'{os.getcwd()}/GUI/assets/static')) != 0:
        for files in glob.glob(f'{os.getcwd()}/GUI/assets/static/*'):
            os.remove(files)
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join('GUI/assets/static', name), "wb") as fp:
        fp.write(base64.decodebytes(data))
    return os.path.join('GUI/assets/static', name)

app.layout = html.Div(id = 'root', children=[
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')])

home_page = html.Div([
    html.H1(children = 'Image Captioning', style={'textAlign': 'center'}),
    html.Div(id = 'right-panel', 
    children = [dcc.Upload(id='upload-image', children = ['Drag and Drop or ', html.A('Select a File')], style={'width': '70%',
                                                                                                                         'height': '60px', 
                                                                                                                         'lineHeight': '60px',
                                                                                                                         'borderWidth': '1px',
                                                                                                                         'borderStyle': 'dashed',
                                                                                                                         'borderRadius': '5px',
                                                                                                                         'textAlign': 'center', 
                                                                                                                         'display': 'block',
                                                                                                                         'margin-left': 'auto',
                                                                                                                         'margin-right': 'auto'},
                                                                                                                         accept='image/*'),
                         html.Br(),
                         html.A(html.Button('Predict', id = 'submit-image', style = {'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}), href='/prediction'),
                         html.Div(id = 'filename', style = {'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'})]
                         )])

predict_page = html.Div(id = 'left-panel', 
                        children = [dcc.Loading(id="loading-2", 
                        children=[html.Div([html.Div(id="output-image")])], type="circle"), 
                        dcc.Link("Return to Homepage", href = '/')])

error_page = html.Div([html.H1("Please upload a file first"), 
                       dcc.Link("Return to Homepage", href = '/')])

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/prediction':
        if len(glob.glob(f'{cwd}/GUI/assets/static/*')) == 0:
            return error_page
        else:
            filepath = glob.glob(f'{cwd}/GUI/assets/static/*')[0]
            test_base64 = base64.b64encode(open(filepath, 'rb').read()).decode('ascii')
            sentence = get_caption(filepath, model_path = "model/best_model (2).pth")
            output = html.Div(children = [html.H2(children = f'{sentence}', style={'textAlign': 'center'} ), 
                                          html.Img(id = 'output-image', src = f'data:image/png;base64,{test_base64}', style={'width': '40%', 'height': 'auto', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'})])
            html_output = html.Div(id = 'left-panel', children = [dcc.Loading(children = [dcc.Link("Return to Homepage", href = '/'), output], id="loading-2", type="circle")])
            return html_output
    else:
        return home_page

@app.callback(Output("filename", "children"), 
             [Input("upload-image", "filename"), 
             Input("upload-image", "contents")])
def update_output(uploaded_filenames, uploaded_file_contents):
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        filepath = save_file(uploaded_filenames, uploaded_file_contents)
        test_base64 = base64.b64encode(open(filepath, 'rb').read()).decode('ascii')
        output = html.Div(children = [html.H6(children = f'File you are submitting: {os.path.basename(filepath)}', style={'textAlign': 'center'} ),
                                      html.Img(id = 'input-image', 
                                               src = f'data:image/png;base64,{test_base64}', 
                                               style={'width': '40%',
                                                      'height': 'auto', 
                                                      'display': 'block',
                                                      'margin-left': 'auto',
                                                      'margin-right': 'auto'})])
        return output

if __name__ == '__main__':
    cwd = os.getcwd()
    if len(os.listdir(f'{cwd}/GUI/assets/static')) != 0:
        for files in glob.glob(f'{cwd}/GUI/assets/static/*'):
            os.remove(files)
    app.run_server(debug=False)