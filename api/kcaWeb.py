from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from filestack import Client
import filestack
import shutil 

from flask import redirect
from os.path import join, dirname, realpath
import os
import pandas as pd
import glob
import numpy as np
import csv 
import prince 
import nltk 
from nltk.tokenize import word_tokenize


# initialize the Flask application
app = Flask(__name__)

#filestack_api_key = 'Aweq3BGqQSZGKRoCOb4Otz'
#filestack_client = Client(filestack_api_key)


app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024
CORPUS_FOLDER = 'files/corpus'
KEYWORD_FOLDER = 'files/keyword_dir'

if not os.path.isdir(CORPUS_FOLDER):
    os.mkdir(CORPUS_FOLDER)

app.config['UPLOAD_FOLDER'] = CORPUS_FOLDER
app.config['KEYWORD_FOLDER'] = KEYWORD_FOLDER

app.config['SECRET_KEY'] = 'kcasecretkey'

class UploadFileForm(FlaskForm):
    corpus = FileField("Corpus File")
    keywords = FileField("Keyword File")
    submit = SubmitField('Submit')



# this is the path to the upload directory
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'csv'}  # allow txt and csv files


@app.route('/',methods = ['GET', 'POST'])
def home():
    
    
    if request.method == 'POST':
        # Save each corpus file
        """
        corpus_files = request.files.getlist("corpus")
        keyword_file = request.files['keywords']

        corpus_file_urls = []
        for file in corpus_files:
            response = filestack_client.upload(file)
            file_url = response.url
            corpus_file_urls.append(file_url) 
        
        keyword_response = filestack_client.upload(keyword_file)
        keyword_file_url = keyword_response.url
        """
        corpus_files = request.files.getlist("corpus")
        	
        for file in corpus_files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #file_url = filestack_client.upload(filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Save the keyword file
        keyword_file = request.files['keywords']
        keyword_filename = secure_filename(keyword_file.filename)
        keyword_file.save(os.path.join(app.config['KEYWORD_FOLDER'], keyword_filename))
        
        # For filestack
        #keyword_file_url = filestack_client.upload(filepath=os.path.join(app.config['KEYWORD_FOLDER'], keyword_filename))
        

        # Directory containing corpus files
        corpus_dir = app.config['UPLOAD_FOLDER']

        keyword_path = os.path.join(app.config['KEYWORD_FOLDER'], keyword_filename)
        
        
        table,  original_reduced_table, reduced_table, row_coordinates, row_contributions, column_coordinates, column_contributions, dimension, avg_ctr, original_reduced_table, original_columns = process_files(app.config['UPLOAD_FOLDER'], keyword_path) #corpus_file_urls,keyword_file_url)

        table_html = table.head(5).to_html()
        #reduced_table_original = or
        reduced_table_html = original_reduced_table.head(5).to_html()

        dimension_html = {}
        for dimension in range(dimension):
            
    # Extract coordinates and contributions for the specific dimension
            dimension_coordinates = column_coordinates.iloc[:, dimension]
            dimension_contributions = column_contributions.iloc[:, dimension]
            
            # Create a DataFrame for the column (keyword) data
            
            dimension_df_col = pd.DataFrame({
                'Keyword': dimension_coordinates.index,
                f'dimension_{dimension+1}': dimension_coordinates.values,
                'ctr': dimension_contributions.values
            })
            dimension_df_col = dimension_df_col[dimension_df_col['ctr'] >= avg_ctr]

            # Write the DataFrame to a CSV file
            #dimension_df_col.to_csv(f'Variables_dimensions/dimension_{dimension+1}_keywords.csv', index=False)

            # For row (text file) data
            row_dimension_coordinates = row_coordinates.iloc[:, dimension]
            row_dimension_contributions = row_contributions.iloc[:, dimension]
            
            indexes = row_dimension_coordinates.index
            fileName = original_reduced_table['FileName'].iloc[indexes]


            
            # Create a DataFrame for the row (text file) data
            dimension_df_row = pd.DataFrame({
                'FileName': fileName,
                f'dimension_{dimension+1}': row_dimension_coordinates.values,
                'ctr': row_dimension_contributions.values
            })


            # Reset the index, just in case
            #dimension_df_row = dimension_df_row.reset_index(drop=True)
            dimension_df_row = dimension_df_row[dimension_df_row['ctr'] >= avg_ctr]

            dimension_df_col.to_csv(f'files/csv/dimension_{dimension+1}_keywords.csv', index=False)
            dimension_df_row.to_csv(f'files/csv/dimension_{dimension+1}_files.csv', index=False)

            dimension_df_col_html = dimension_df_col.to_html()
            dimension_df_row_html = dimension_df_row.to_html()
            dimension_html[f'dimension_{dimension+1}'] = {'col': dimension_df_col_html, 'row': dimension_df_row_html}

            table.to_csv('files/csv/table.csv', index=False)
            original_reduced_table.to_csv('files/csv/reduced_table.csv', index=False)

        return render_template('index.html', table_html=table_html, reduced_table_html=reduced_table_html,dimension_html=dimension_html)
    


    return render_template('index.html', table_html="", reduced_table_html="", dimension_html={})


def delete_files():
    shutil.rmtree(CORPUS_FOLDER)
    shutil.rmtree(KEYWORD_FOLDER)
    shutil.rmtree('files/csv')
    os.mkdir(CORPUS_FOLDER)
    os.mkdir(KEYWORD_FOLDER)
    os.mkdir('files/csv')


@app.route('/end_session', methods=['POST'])
def end_session():
    delete_files()
    return redirect(url_for('home')) 


@app.route('/download/<path:filename>', methods=['GET'])
def download(filename):
    """This route serves the files to be downloaded."""
    print(f"Filename: {filename}")
    return send_from_directory('files/csv', filename, as_attachment=True)


"""	
@app.route('/end-session', methods=['POST'])
def end_session():
    corpus_file_urls = request.form.getlist("corpus_file_urls")
    keyword_file_url = request.form.get("keyword_file_url")

    # Delete the files from Filestack
    for file_url in corpus_file_urls:
        filestack_client.delete(file_url)
    filestack_client.delete(keyword_file_url)

    # Redirect to the home page
    return redirect('/')
"""

def process_files(corpus_dir, keyword_path):

    """
    corpus = []
    file_names = []
    for file_url in corpus_file_urls:
        file = request.urlopen(file_url)
        text = file.read().decode('utf-8')
        corpus.append(text)
        file_names.append(file_url)  # Use the file URL as the file name, adjust as needed
    corpus_df = pd.DataFrame({'Text': corpus, 'FileName': file_names})

    # Read the keyword file from Filestack
    keyword_file = request.urlopen(keyword_file_url)
    keyword_text = keyword_file.read().decode('utf-8')
    keywords = pd.read_csv(pd.compat.StringIO(keyword_text))

    """
    # Step 1: Read in target corpus
    file_paths = glob.glob(os.path.join(corpus_dir, '*.txt'))
    corpus = []
    file_names = []
    for file_path in file_paths:
        if os.path.splitext(file_path)[1] == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                corpus.append(text)
                file_names.append(os.path.basename(file_path))
    corpus_df = pd.DataFrame({'Text': corpus, 'FileName': file_names})

    # Step 2: Read in keyword list
    keyword_extension = keyword_path.split('.')[-1].lower()

    if keyword_extension == 'csv':
        keywords = pd.read_csv(keyword_path)
    else:
        keywords = pd.read_csv(keyword_path, sep='\t', names=['Keyword'])
    
    # Step 3: Search and record presence/absence of keywords in the corpus
    data = []
    for idx, row in corpus_df.iterrows():
        keyword_presence_row = {'FileName': row['FileName']}  
        for keyword in keywords['Keyword']:
            if keyword in row['Text']:
                keyword_presence_row[keyword] = 'P'
            else:
                keyword_presence_row[keyword] = 'A'
        data.append(keyword_presence_row)

    table = pd.DataFrame(data)  

    threshold = int(0.05 * len(corpus_df))
    keyword_columns = [col for col in table.columns if col != 'FileName']  # Get only keyword columns
    reduced_table = table.loc[:, ['FileName'] + [col for col in keyword_columns if table[col].value_counts().get('P', 0) > threshold]]

    corpus_df['WordCount'] = corpus_df['Text'].apply(lambda x: len(word_tokenize(x)))
    reduced_table['Quantitative Supplementary Variable'] = corpus_df['WordCount']

    original_reduced_table = reduced_table
    reduced_table = reduced_table.iloc[:, 1:-1]
    original_columns = reduced_table.columns.tolist() 

    #reduced_table.to_csv('reduced_table.csv', index=False)

    mca_df = prince.MCA(
        n_components=10,
    )
    mca_df = mca_df.fit(reduced_table)

    avg_ctr = 100 / (len(reduced_table.columns) * 2)
    row_coordinates = mca_df.row_coordinates(reduced_table)
    row_contributions = mca_df.row_contributions_*100
    column_coordinates = mca_df.column_coordinates(reduced_table)
    column_contributions = mca_df.column_contributions_*100


    dimensions = mca_df.n_components

    # In the end, return the generated DataFrames
    return table, original_reduced_table, reduced_table, row_coordinates, row_contributions, column_coordinates, column_contributions, dimensions, avg_ctr, original_reduced_table, original_columns

if __name__ == '__main__':
    app.run(debug = True)
