# PdfScholar

PdfScholar is an AI-powered web tool designed to revolutionize how students interact with their educational resources. With PdfScholar, users can upload PDFs—whether they’re textbooks, research papers, or any academic materials—and engage in a dynamic conversation with the content. Ask questions directly related to the text, and receive accurate, context-aware answers as if you're conversing with the PDFs themselves.

This tool is ideal for students at any academic level, offering personalized assistance in understanding complex concepts, extracting key information, or even summarizing research findings. PdfScholar is your academic companion, making learning more interactive and efficient by focusing exclusively on the materials you provide.

## Features

- **Upload PDFs**: Support for textbooks, research papers, articles, and other academic materials.
- **Ask Questions**: Interact with the content, ask context-aware questions, and get precise answers.
- **Summarization**: Quickly extract key points or generate concise summaries of large sections.
- **Easy Navigation**: Find information within large documents quickly with smart indexing.
- **Personalized Learning**: Tailored assistance based on the specific materials you provide, making it ideal for any academic level.

## Technology Stack

- **Frontend**: streamlit
- **Backend**: Python 
- **AI Models**: OpenAI's GPT-40 mini
- **File Handling**: PDF processing with `PyPDF2`
- **Deployment**: streamlit

## Getting Started

### Prerequisites

- Python 3.x
- Required Python Libraries: `PyPDF2`, `shelve`, `openai`, `streamlit`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pdfscholar.git
   cd pdfscholar
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Run the application:
   ```bash
   run streamlit ./src/app.py

### Usage

Once the application is up and running, follow these steps to interact with PdfScholar:

1. **Upload a PDF Document**
   - Click on the "Upload" button to upload any academic PDF (textbooks, research papers, articles, etc.).

2. **Ask Questions**
   - After the PDF is processed, type a question related to the content in the chat box.
   - Examples of questions you can ask:
     - "Summarize this section."
     - "What is the main argument in Chapter 3?"
     - "Explain the formula on page 10."
     - "What is the definition of a key term?"

3. **Receive Context-Aware Answers**
   - PdfScholar will analyze the content of the uploaded PDF and provide accurate, context-based answers or summaries related to your question.
   
4. **Navigate Through the Document**
   - Use PdfScholar’s smart navigation to jump to different sections of the document efficiently and locate relevant information with ease.
   
5. **Summarization**
   - If needed, you can request a summary of a specific section or the entire document by asking PdfScholar to "Summarize Chapter 1" or "Summarize the whole document."


