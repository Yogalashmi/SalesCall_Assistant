import os
from groq import Groq
from langchain.llms import Cohere
import re
class APIClient:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)

    def extract_clothing_questions(self, text):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": f"""
                    Extract only product-related questions (specifically about clothing and garments) from the following text. Analyze the customer's questions to identify their major concerns or pain points regarding the product. Focus on the product attributes or features that the customer is most concerned about, and explain these key concerns clearly and concisely. If there are no product-related questions, return 'Nothing'.
                    Additionally, analyze the overall emotion of the customer, such as happy, angry, frustrated... and describe how their emotion may have transformed throughout the interaction. For example, if a customer is frustrated, did they become happy? Present the transformation as: frustrated --> confused --> happy.
                    Finally, assess whether the customer is overall satisfied or dissatisfied at the end of the interaction. Strictly follow the output structure; don't add any explanation of why you provided the response.

                    Output Structure:
                        1. List of Product-Related Questions:
                             - [Question 1]
                             - [Question 2]
                             - ...

                        2. Overall Emotion Transformation:
                             - [Initial Emotion] --> [Intermediate Emotion] --> [Final Emotion]

                        3. Overall Satisfaction:
                             - [Satisfied/Not Satisfied]

                    \n{text}
                    """
                }],
                model="llama-3.1-70b-versatile",
            )
            response = chat_completion.choices[0].message.content
            questions_match = re.search(r"1\. List of Product-Related Questions:(.*?)2\.", response, re.DOTALL)
            questions = questions_match.group(1).strip().split('\n') if questions_match else []

            # Extract overall emotion transformation and satisfaction
            rest_of_response = response.replace(questions_match.group(0), '').strip() if questions_match else response

            return questions, rest_of_response
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def customer_insights(self, text):
        chat_completion = self.client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"""
                Extract customer concerns, issues faced, feedback from the customer side, and initial needs/interests from the following call transcript. 
                Please analyze the customer's statements and identify the major concerns or pain points they have regarding the product (clothing and garments). 
                Focus on the product attributes or features the customer is most concerned about. Check if they have provided any feedback regarding the shop or the product or the service the company is providing. 
                Explain the key concerns in a clear, concise way. If there are none, return " ".

                Context: 
                {text}

                Output Format:
                - **Customer Concerns**:
                  - [List of concerns]
                  
                - **Issues Faced**:
                  - [List of issues]
                  
                - **Customer Feedback**:
                  - [List of feedback, if mentioned]
                  
                - **Initial Needs/Interests**:
                  - [List of needs/interests]
                """
            }],
            model="llama-3.1-70b-versatile",
        )
        return chat_completion.choices[0].message.content
# Initialize the APIClient
try:
    
    groq_api_key = os.getenv('GROQ_API_KEY')
    api_client = APIClient(groq_api_key)
    print("APIClient initialized successfully.")
    clothing_questions, rest_of_response =api_client.extract_clothing_questions("i wnat a neautful summer dress?")
    print(clothing_questions)
    print(rest_of_response)
except Exception as e:
    print(f"Error initializing APIClient: {str(e)}")