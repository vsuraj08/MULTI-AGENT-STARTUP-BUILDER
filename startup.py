import streamlit as st
import google.generativeai as genai
from agno.agent import Agent
from agno.models.google import Gemini
import time
from dotenv import load_dotenv
import os
import graphviz
import plotly.express as px
import pandas as pd

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    st.error("Google API key not found. Please add it to the `.env` file.")
    st.stop()

# Initialize Agents
idea_enhancer_agent = Agent(
    name="Idea Enhancer Agent",
    role="Refine and enhance the user's startup idea into a comprehensive description.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions=[
        "Analyze the user's startup idea and provide a detailed, refined description.",
        "Include potential market opportunities, target audience, and unique value proposition.",
    ],
    markdown=True,
)

workflow_developer_agent = Agent(
    name="Workflow Developer Agent",
    role="Develop a step-by-step implementation plan for the startup idea.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions=[
        "Create a detailed, actionable workflow for implementing the startup idea.",
        "Break down the plan into phases with clear milestones and deliverables.",
    ],
    markdown=True,
)

resource_analyst_agent = Agent(
    name="Resource Analyst Agent",
    role="Identify the necessary resources, skill sets, and budget for the startup.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions=[
        "List the required resources (e.g., tools, software, hardware).",
        "Identify the skill sets needed for the team.",
        "Provide an estimated budget for the project.",
    ],
    markdown=True,
)

recruitment_content_creator_agent = Agent(
    name="Recruitment Content Creator Agent",
    role="Generate job postings for the roles required to execute the startup idea.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions=[
        "Create detailed job descriptions for each role.",
        "Include responsibilities, qualifications, and desired skills.",
    ],
    markdown=True,
)

interview_question_designer_agent = Agent(
    name="Interview Question Designer Agent",
    role="Craft tailored interview questions for evaluating candidates.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions=[
        "Design interview questions specific to each role.",
        "Include technical, behavioral, and situational questions.",
    ],
    markdown=True,
)

evaluation_agent = Agent(
    name="Evaluation Agent",
    role="Validate the feasibility of the startup idea.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions=[
        "Analyze the market size, competition, and potential challenges.",
        "Provide data-driven insights to validate the idea.",
    ],
    markdown=True,
)

visualization_agent = Agent(
    name="Visualization Agent",
    role="Create interactive charts and graphs for budgets, timelines, and resource allocation.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions=[
        "Visualize the workflow as a flowchart.",
        "Create charts for budgets and resource allocation.",
        "Generate well-formatted tables for the data.",
    ],
    markdown=True,
)

feedback_agent = Agent(
    name="Feedback Agent",
    role="Collect user feedback and refine agent outputs dynamically.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions=[
        "Ask the user for feedback on each step.",
        "Refine the outputs based on user feedback.",
    ],
    markdown=True,
)

export_agent = Agent(
    name="Export Agent",
    role="Export the generated plans to external tools.",
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions=[
        "Export the workflow to Google Sheets, Trello, or Notion.",
        "Post job openings to LinkedIn or Indeed.",
    ],
    markdown=True,
)

# Helper function to handle streaming responses
def display_streaming_response(response_generator, placeholder):
    full_response = ""
    for chunk in response_generator:
        if hasattr(chunk, "content"):  # Check if the chunk has the 'content' attribute
            full_response += chunk.content
            placeholder.markdown(full_response)  # Update the response in real-time
        time.sleep(0.1)  # Simulate streaming delay
    return full_response

# Helper function to create a flowchart
def create_flowchart(workflow):
    dot = graphviz.Digraph()
    steps = workflow.split("\n")
    for i, step in enumerate(steps):
        dot.node(str(i), step)
        if i > 0:
            dot.edge(str(i-1), str(i))
    return dot

# Streamlit App
st.title("Advanced Startup Builder Platform")
st.write("This app helps you turn your startup idea into actionable steps, resources, and recruitment plans.")

# User Input
user_idea = st.text_area("Enter your startup idea (e.g., 'I want to build a platform for AI-powered education tools'):")

if user_idea:
    with st.spinner("Processing your startup idea..."):
        try:
            # Step 1: Enhance the Idea
            st.write("### Step 1: Enhanced Startup Idea")
            enhanced_idea_placeholder = st.empty()
            enhanced_idea_response = idea_enhancer_agent.run(user_idea, stream=True)
            enhanced_idea = display_streaming_response(enhanced_idea_response, enhanced_idea_placeholder)

            # Step 2: Validate the Idea
            st.write("### Step 2: Idea Validation")
            validation_placeholder = st.empty()
            validation_response = evaluation_agent.run(enhanced_idea, stream=True)
            validation = display_streaming_response(validation_response, validation_placeholder)

            # Step 3: Develop Workflow
            st.write("### Step 3: Implementation Workflow")
            workflow_placeholder = st.empty()
            workflow_response = workflow_developer_agent.run(enhanced_idea, stream=True)
            workflow = display_streaming_response(workflow_response, workflow_placeholder)

            # Step 4: Identify Resources
            st.write("### Step 4: Required Resources and Budget")
            resources_placeholder = st.empty()
            resources_response = resource_analyst_agent.run(workflow, stream=True)
            resources = display_streaming_response(resources_response, resources_placeholder)

            # Step 5: Create Job Postings
            st.write("### Step 5: Job Postings")
            job_postings_placeholder = st.empty()
            job_postings_response = recruitment_content_creator_agent.run(resources, stream=True)
            job_postings = display_streaming_response(job_postings_response, job_postings_placeholder)

            # Step 6: Design Interview Questions
            st.write("### Step 6: Interview Questions")
            interview_questions_placeholder = st.empty()
            interview_questions_response = interview_question_designer_agent.run(job_postings, stream=True)
            interview_questions = display_streaming_response(interview_questions_response, interview_questions_placeholder)

            # Step 7: Visualize the Plan
            st.write("### Step 7: Visualization")
            visualization_placeholder = st.empty()
            visualization_response = visualization_agent.run(workflow, stream=True)
            visualization = display_streaming_response(visualization_response, visualization_placeholder)

            # Create a flowchart
            st.write("#### Flowchart")
            dot = create_flowchart(workflow)
            st.graphviz_chart(dot)

            # Download flowchart as .dot file
            st.download_button(
                label="Download Flowchart (DOT)",
                data=dot.source,
                file_name="workflow.dot",
                mime="text/plain",
            )

            # Create a budget chart
            st.write("#### Budget Chart")
            budget_data = {"Category": ["Development", "Marketing", "Operations"], "Amount": [50000, 30000, 20000]}
            df = pd.DataFrame(budget_data)
            fig = px.bar(df, x="Category", y="Amount", title="Project Budget")
            st.plotly_chart(fig)

            # Download budget data as CSV
            st.download_button(
                label="Download Budget Data (CSV)",
                data=df.to_csv(index=False),
                file_name="budget.csv",
                mime="text/csv",
            )

            # Step 8: Collect Feedback
            st.write("### Step 8: Feedback")
            feedback_placeholder = st.empty()
            feedback_response = feedback_agent.run("Please provide feedback on the generated plan.", stream=True)
            feedback = display_streaming_response(feedback_response, feedback_placeholder)

            # Step 9: Export the Plan
            st.write("### Step 9: Export the Plan")
            export_placeholder = st.empty()
            export_response = export_agent.run("Export the plan to Google Sheets.", stream=True)
            export = display_streaming_response(export_response, export_placeholder)

            # Final Output
            st.success("Startup plan generated successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")