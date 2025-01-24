import pdfkit
import base64
import streamlit as st
def generate_pdf(content, pdf_file_name):
    html_content = f"""
    <html>
        <head>
            <title>Report</title>
        </head>
        <body>
            {content}
        </body>
    </html>
    """

    pdfkit_config = pdfkit.configuration(wkhtmltopdf='C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe')
    try:
        pdfkit.from_string(html_content, pdf_file_name, configuration=pdfkit_config)
        with open(pdf_file_name, "rb") as pdf_file:
            pdf_data = pdf_file.read()
        return pdf_data
    except Exception as e:
        st.error(f"An error occurred while generating the PDF: {e}")
        return None