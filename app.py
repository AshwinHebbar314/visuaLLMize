# %%
import csv
import json
import os
import shutil

import envs
import google.generativeai as genai
import matplotlib
import matplotlib.pyplot as plt
from dotenv import load_dotenv

matplotlib.use("agg")
import pandas as pd
import seaborn as sns
from flask import Flask, render_template, request

load_dotenv()


class Visualize:
    # Takes in a dictionary of visualizations and saves the sns plots to a folder
    # All possible vizualizations suggestions come in this format:

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)

    def save_plots(self, visualization_suggestions):
        # set figure size
        plt.figure(figsize=(12, 8))
        plt.xticks(rotation=45)

        for visualization in visualization_suggestions:
            data = self.data
            chart_type = visualization["chart_type"]
            if chart_type == "scatterplot":
                c = sns.scatterplot(
                    x=visualization["x_column"],
                    y=visualization["y_column"],
                    hue=visualization["hue_column"],
                    data=self.data,
                )
                # #

            elif chart_type == "lineplot":
                if len(self.data[visualization["x_column"]].unique()) < 10:
                    c = sns.lineplot(
                        x=visualization["x_column"],
                        y=visualization["y_column"],
                        hue=visualization["hue_column"],
                        data=self.data,
                    )
                    # #

                else:
                    # display only the top 10 unique values
                    top_10 = (
                        self.data[visualization["x_column"]]
                        .value_counts()
                        .head(10)
                        .index
                    )

                    data = self.data[self.data[visualization["x_column"]].isin(top_10)]
                    c = sns.lineplot(
                        x=visualization["x_column"],
                        y=visualization["y_column"],
                        hue=(
                            visualization["hue_column"]
                            if "hue_column" in visualization
                            else None
                        ),
                        data=data,
                    )
                    #

            elif chart_type == "barplot":
                if len(self.data[visualization["x_column"]].unique()) < 10:
                    c = sns.barplot(
                        x=visualization["x_column"],
                        y=visualization["y_column"],
                        hue=visualization["hue_column"],
                        data=self.data,
                    )
                    #

                else:
                    # display only the top 10 unique values
                    top_10 = (
                        # reverse sort the values by the count and get the top 10
                        self.data[visualization["x_column"]]
                        .value_counts()
                        .head(10)
                        .index
                    )
                    #

                    data = self.data[self.data[visualization["x_column"]].isin(top_10)]
                    c = sns.barplot(
                        x=visualization["x_column"],
                        y=visualization["y_column"],
                        hue=visualization["hue_column"],
                        data=data,
                    )
                    #

            elif chart_type == "histplot":
                c = sns.histplot(
                    x=visualization["x_column"],
                    bins=visualization["bins"],
                    data=self.data,
                )
                #

            elif chart_type == "boxplot":
                if len(self.data[visualization["x_column"]].unique()) < 10:
                    c = sns.boxplot(
                        x=visualization["x_column"],
                        y=visualization["y_column"],
                        hue=visualization["hue_column"],
                        data=self.data,
                    )
                    #

                else:
                    # display only the top 10 unique values
                    top_10 = (
                        self.data[visualization["x_column"]]
                        .value_counts()
                        .head(10)
                        .index
                    )

                    data = self.data[self.data[visualization["x_column"]].isin(top_10)]
                    c = sns.boxplot(
                        x=visualization["x_column"],
                        y=visualization["y_column"],
                        hue=visualization["hue_column"],
                        data=data,
                    )
                    #

            elif chart_type == "violinplot":
                if len(self.data[visualization["x_column"]].unique()) < 10:
                    c = sns.violinplot(
                        x=visualization["x_column"],
                        y=visualization["y_column"],
                        hue=visualization["hue_column"],
                        data=self.data,
                    )
                    #

                else:
                    # display only the top 10 unique values
                    top_10 = (
                        self.data[visualization["x_column"]]
                        .value_counts()
                        .head(10)
                        .index
                    )

                    data = self.data[self.data[visualization["x_column"]].isin(top_10)]
                    c = sns.violinplot(
                        x=visualization["x_column"],
                        y=visualization["y_column"],
                        hue=visualization["hue_column"],
                        data=data,
                    )
                    #

            elif chart_type == "heatmap":
                # generate a correlation matrix and plot the heatmap
                try:
                    # discard all non-numerical columns
                    new_data = self.data.select_dtypes(include=["float64", "int64"])
                    corr = new_data.corr()
                    c = sns.heatmap(corr, cmap=visualization["cmap"])
                    #

                except Exception as e:
                    print("Error: ", e)

            elif chart_type == "kdeplot":
                c = sns.kdeplot(
                    x=visualization["x_column"],
                    y=visualization["y_column"],
                    hue=visualization["hue_column"],
                    data=self.data,
                )
                #

            elif chart_type == "regplot":
                c = sns.regplot(
                    x=visualization["x_column"],
                    y=visualization["y_column"],
                    data=self.data,
                )
                #

            elif chart_type == "jointplot":
                c = sns.jointplot(
                    x=visualization["x_column"],
                    y=visualization["y_column"],
                    kind=visualization["kind"],
                    data=self.data,
                )
                #

            elif chart_type == "swarmplot":
                c = sns.swarmplot(
                    x=visualization["x_column"],
                    y=visualization["y_column"],
                    hue=visualization["hue_column"],
                    data=self.data,
                )
                #

            else:
                print("Invalid chart type")

            # Save the plot as plotname_index.png

            plt.savefig(
                "static/plots/"
                + f"{chart_type}_{visualization_suggestions.index(visualization)}.png"
            )
            plt.clf()


# set the environment variables
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 100000,
    "response_mime_type": "application/json",
}

system_instructions = {
    # "meta_summarize": """You are a data summarization expert. You are given a dataset and asked to summarize the data. Based on the file provided, give some basic information about the dataset. You are expected to only return a markdown output based on the data provided. You are NOT expected to create the visualization itself. Or provide Code of any sort. Keep the summary in english only.""",
    "meta_summarize_and_get_viz_instructions": """You play two roles in this task. You will take roles accordingly based on the "summary" or "visualization_suggestions" prompt you receive.
    
    Firstly, You are a data summarization expert. You are given a dataset and asked to summarize the data. Based on the file provided, give some basic information about the dataset. You are expected to only return a markdown output based on the data provided. You are NOT expected to create the visualization itself. Or provide Code of any sort. Keep the summary in english only. Possible things you can include are:
    1. A list of columns, a quick explanation and their data types.
    2. The number of rows and columns in the dataset

    among others.
    
    Second, You are a data visualization expert. You are given a dataset and asked to create a visualization that best represents the data. You are free to choose the type of visualization you think is most appropriate for the data subject to the conatraints given below. You are expected to ONLY return a json output based on the data provided. You are NOT expected to create the visualization itself. You are given the following constraints:
  1. You can suggest upto 8 different types of visualizations based on the data provided. If really required, more than 8 visualizations can be suggested, but only if it is absolutely necessary.
  2. The visualizations should be based on the data provided.
  3. Maks sure to use only the attributes present in the data for the visualizations. don't hallucinate any new attributes.
  3. The output should be in json format. Use this JSON Schema for the output, and output the JSON as a string. nothing else should be returned. make sure to follow the schema and only return the visualizations mentioned in the schema, nothing outside of it. If a visualization technique is not applicable to the data, you can skip it.:
  {
    "summary": "This is a placeholder string, replace this string with the summary of the data in markdown format.",
    "visualization_suggestions": [
        {
            "chart_type": "scatterplot",
            "x_column": "string",
            "y_column": "string",
            "hue_column": "string"
        },
        {
            "chart_type": "lineplot",
            "x_column": "string",
            "y_column": "string",
            "hue_column": "string"
        },
        {
            "chart_type": "barplot",
            "x_column": "string",
            "y_column": "string",
            "hue_column": "string"
        },
        {
            "chart_type": "histplot",
            "x_column": "string",
            "bins": "integer"
        },
        {
            "chart_type": "boxplot",
            "x_column": "string",
            "y_column": "string",
            "hue_column": "string"
        },
        {
            "chart_type": "violinplot",
            "x_column": "string",
            "y_column": "string",
            "hue_column": "string"
        },
        {
            "chart_type": "heatmap",
            "data": "2D array or DataFrame",
            "cmap": "string"
        },
        {
            "chart_type": "kdeplot",
            "x_column": "string",
            "y_column": "string",
            "hue_column": "string"
        },
        {
            "chart_type": "regplot",
            "x_column": "string",
            "y_column": "string"
        },
        {
            "chart_type": "jointplot",
            "x_column": "string",
            "y_column": "string",
            "kind": "string"
        },
        {
            "chart_type": "swarmplot",
            "x_column": "string",
            "y_column": "string",
            "hue_column": "string"
        }
    ]
}

Keep in mind to STRICTLY follow the constraints given above. If you do not follow the constraints, your output will be rejected and you will be penalized.
  """,
}


summarize_and_viz_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-exp-0827",
    generation_config=generation_config,
    system_instruction=system_instructions["meta_summarize_and_get_viz_instructions"],
)

# create a Flask app
app = Flask(__name__)

# enable debug mode
app.config["DEBUG"] = True
app.config["FLASK_ENV"] = "development"
app.config["UPLOAD_FOLDER"] = "uploads"


def csv_to_string(filename):
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        csv_string = ""
        for row in csv_reader:
            csv_string += ",".join(row) + "\n"
        return csv_string


@app.route("/flush", methods=["POST"])
def flush():
    # delete all files in the uploads folder using shutil
    shutil.rmtree(app.config["UPLOAD_FOLDER"])
    os.mkdir(app.config["UPLOAD_FOLDER"])

    # delete all files in the static/plots folder using shutil
    shutil.rmtree("static/plots")
    os.mkdir("static/plots")

    return "Files Deteted"


@app.route("/")
def hello_world():
    return render_template("main.html")


@app.route("/visualize", methods=["POST"])
def visualize():
    prompt = request.form["prompt"]

    # get the dataset from the form and save it.
    dataset = request.files["dataset"]
    dataset.save(os.path.join(app.config["UPLOAD_FOLDER"], dataset.filename))

    print("Prompt: ", prompt)
    print("Dataset: ", dataset.filename)

    # upload_file = genai.upload_file("uploads/" + dataset.filename)
    # print("Upload file: ", upload_file)

    upload_file = csv_to_string("uploads/" + dataset.filename)

    # pass the prompt and the dataset to the model
    viz_response = summarize_and_viz_model.generate_content(
        [prompt, upload_file],
        # system_instruction=system_instructions["get_viz_instructions"],
        safety_settings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        ],
    )

    viz_dict = json.loads(viz_response.text)
    print(type(viz_dict))
    print("Visualization suggestions: ", viz_dict["visualization_suggestions"])

    # save the visualizations to a folder
    Visualize("uploads/" + dataset.filename).save_plots(
        viz_dict["visualization_suggestions"]
    )

    metadata_info = viz_dict["summary"]

    print(metadata_info)

    # return "Visualization suggestions: " + viz_response.text
    return render_template(
        "visualize.html", images=os.listdir("static/plots"), metadata_info=metadata_info
    )


if __name__ == "__main__":
    app.run(debug=True)
