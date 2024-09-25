# VisuaLLMize: AI-Powered Data Visualization

**VisuaLLMize** is a web application that leverages the power of Large Language Models (LLMs) to automatically generate insightful visualizations from your data. Simply upload a CSV file and provide a brief prompt describing your desired visualization, and VisuaLLMize will handle the rest!

## Features

* **Automated Visualization Generation:** VisuaLLMize utilizes Google's Gemini Pro LLM to intelligently suggest and generate a variety of visualizations based on your data and prompt.
* **Support for Various Chart Types:**  The application supports a wide range of chart types, including scatterplots, line plots, bar plots, histograms, box plots, violin plots, heatmaps, and more.
* **User-Friendly Interface:** The intuitive web interface makes it easy to upload data, provide prompts, and view the generated visualizations.
* **Data Summarization:** VisuaLLMize provides a concise summary of the uploaded dataset, offering valuable insights into the data's characteristics.
* **Easy Integration:** The project is built using Flask, making it easy to deploy and integrate into existing workflows.

## How it Works

1. **Upload Dataset:** Upload your data in CSV format through the web interface.
2. **Provide Prompt:** Describe the desired visualization using a natural language prompt. For example, "Show the relationship between sales and advertising spend" or "Visualize the distribution of customer ages."
3. **Generate Visualizations:** VisuaLLMize processes your data and prompt, utilizing the Gemini Pro LLM to suggest and generate relevant visualizations.
4. **View and Analyze:** The generated visualizations are displayed on the web interface, allowing you to explore and analyze your data effectively.

## Technologies Used

* **Flask:** Python web framework for building the application.
* **Google Generative AI:** Provides access to Google's Gemini Pro LLM for visualization generation and data summarization.
* **Seaborn & Matplotlib:** Python libraries for creating visually appealing and informative visualizations.
* **HTML, CSS, & JavaScript:** Front-end technologies for building the user interface.

## Installation & Setup

1. **Clone the repository:** `git clone https://github.com/AshwinHebbar314/VisuaLLMize`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Set up Google Generative AI API Key:** Obtain an API key from Google Cloud Console and replace `YOUR API KEY` in `app.py`.
4. **Run the application:** `python app.py`

## Usage

1. Open the application in your web browser (typically at `http://127.0.0.1:5000/`).
2. Upload your CSV dataset.
3. Enter a prompt describing the desired visualization.
4. Click "Submit" to generate the visualizations.
5. View the generated visualizations and data summary on the results page.

## Future Enhancements

* **Interactive Visualizations:** Implement interactive features to enable users to explore the data further.
* **Support for More Data Formats:** Expand support for other data formats beyond CSV.
* **Customization Options:** Provide more options for customizing the appearance and style of the visualizations.
* **Integration with Data Analysis Tools:** Integrate with popular data analysis tools to provide a seamless workflow.

## Disclaimer

This project is intended for demonstration and educational purposes. The generated visualizations are based on the capabilities of the underlying LLM and may not always be optimal or accurate. It is recommended to validate the insights obtained from the visualizations with other data analysis techniques.
