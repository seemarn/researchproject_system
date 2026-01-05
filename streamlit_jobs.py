import sys
import streamlit as st

st.write("Python OK")

try:
    import spacy
    st.success("spaCy loaded successfully")
except Exception as e:
    st.error("spaCy import failed")
    st.exception(e)


import streamlit as st
import spacy
import pandas as pd
from collections import Counter
import re
import ast
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Career Match",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS matching the prototype design
st.markdown("""
<style>
    /* Import custom font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
    /* Apply the font globally */
    html, body, [class*="css"], div, p, span, label, input, textarea, button, h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
    }
    

    /* Global styles */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .header-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 400;
    }
    
    /* Input card */
    .input-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }
    
    .section-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    
    /* Skills display */
    .skills-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }
    
    .skill-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Job card */
    .job-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }
    
    .job-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .job-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .job-company {
        font-size: 1rem;
        color: #7f8c8d;
        margin-bottom: 0.5rem;
    }
    
    .match-score {
        display: inline-block;
        background: #2ecc71;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .matched-skills {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #ecf0f1;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        width: 100%;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        font-size: 0.95rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load NER model
@st.cache_resource
def load_ner_model():
    try:
        nlp = spacy.load("./skill_ner_model")
        return nlp
    except Exception as e:
        st.error(f"Error loading NER model: {str(e)}")
        st.info("Make sure the 'skill_ner_model' directory is in the same folder as app.py")
        return None

# Load job data in local
# @st.cache_data
# def load_jobs(file_path="data/jobstreet_all_job_dataset_2025_skills_ner_clustered.csv"):
#     df = pd.read_csv(file_path)
#     return df

# Load job data hugging face
@st.cache_data
def load_jobs(file_path="https://huggingface.co/datasets/seemarn/jobstreet/resolve/main/jobstreet_all_job_dataset_2025_skills_ner_clustered.csv"):
    df = pd.read_csv(file_path)
    return df

# Extract skills using NER model
def extract_skills(text, nlp):
    if nlp is None:
        return []
    
    doc = nlp(text)
    skills = [ent.text.strip() for ent in doc.ents if ent.label_ == "SKILL"]
    # Remove duplicates and normalize
    skills = list(set([skill.lower().strip() for skill in skills]))
    return skills


# Calculate match score using Cosine Similarity
def calculate_match_score(user_skills, job_skills):
    if not job_skills or not user_skills:
        return 0, set()
    
    # 1. Normalize lists (Lowercase + Strip)
    # We keep them as LISTS, we do NOT join them into sentences yet
    user_skills = [s.lower().strip() for s in user_skills]
    job_skills = [s.lower().strip() for s in job_skills]
    
    # 2. Find intersection for display
    user_set = set(user_skills)
    job_set = set(job_skills)
    matches = user_set.intersection(job_set)
    
    if not matches:
        return 0, matches

    # 3. Configure Vectorizer to respect Lists (Phrases) instead of Words
    # tokenizer=lambda x: x tells Vectorizer "I already gave you a list of tokens, don't split them"
    # preprocessor=lambda x: x tells Vectorizer "Don't lowercase/strip again, I already did it"
    vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
    
    # 4. Create Vectors
    # We pass the lists directly: [[skill, skill], [skill, skill]]
    vectors = vectorizer.fit_transform([user_skills, job_skills]).toarray()
    
    # 5. Calculate Cosine Similarity
    similarity_matrix = cosine_similarity(vectors)
    score = similarity_matrix[0][1] * 100
    
    return score, matches

# Main app
def main():
    # Header
    st.markdown("""
        <div class="header-container">
            <div class="header-icon">🎯</div>
            <div class="header-title">Career Match</div>
            <div class="header-subtitle">Discover your perfect job match by analyzing your skills and experience with our intelligent recommendation system</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load resources
    nlp = load_ner_model()
    jobs_df = load_jobs()
    
    if nlp is None or jobs_df is None:
        st.stop()
    
    # Input section
    st.markdown('', unsafe_allow_html=True)
    st.markdown('<h3><b>⚙️Describe Your Skills</b></h3>', unsafe_allow_html=True)
    st.markdown('Tell us about your professional skills, experience, and expertise. Be as detailed as possible.', 
                unsafe_allow_html=True)
    
    # Example text for demonstration
    example_text = """I have 5 years of experience in React and TypeScript development. I'm skilled in building responsive web applications, working with REST APIs, and using modern development tools like Git and Docker. I also have experience with database design and SQL, particularly with PostgreSQL and MongoDB. I've worked extensively with Node.js for backend development and have strong problem-solving skills. I enjoy collaborating with cross-functional teams and have led small development teams on several projects. Additionally, I have experience with AWS cloud services and CSS frameworks like Tailwind CSS."""
    
    if "profile_text" not in st.session_state:
        st.session_state.profile_text = ""

    user_input = st.text_area(
        "Your Skills and Experience",
        key="profile_text",
        height=200,
        placeholder="Enter your skills, experience, and expertise here...",
        value="",
        label_visibility="collapsed"
    )
    
    def fill_example():
        st.session_state.profile_text = example_text

    col1, col2 = st.columns([4, 1])
    with col1:
        analyze_button = st.button("🔍 Analyze Skills & Find Jobs")
    with col2:
        st.button("📝 Try Example", on_click=fill_example)

    
    st.markdown('', unsafe_allow_html=True)
    user_input = st.session_state.profile_text

    
    # Process and display results
    if analyze_button and user_input:
        with st.spinner("Analyzing your skills..."):
            # Extract skills
            extracted_skills = extract_skills(user_input, nlp)
            
            if not extracted_skills:
                st.warning("No skills detected. Please provide more details about your technical skills and experience.")
                return
            
            # Display extracted skills
            st.markdown('', unsafe_allow_html=True)
            st.markdown(f'<h3><b>✅ Extracted Skills</b></h3>', unsafe_allow_html=True)
            st.markdown(f'We\'ve identified {len(extracted_skills)} skills from your description', 
                        unsafe_allow_html=True)
            badge_css = """
            <style>
            .skill-badge{
                display:inline-block;
                margin:4px 6px 0 0;
                padding:6px 10px;
                border-radius:12px;
                background:#0ea5e9; color:#fff; font-weight:600;
                white-space:nowrap;
            }
            </style>
            """
            badges_html = "".join(f"<span class='skill-badge'>{s}</span>" for s in extracted_skills)
            st.markdown(badge_css + badges_html, unsafe_allow_html=True)
            #skills_html = ", ".join([f'{skill.title()}' for skill in extracted_skills])
            #st.markdown(skills_html, unsafe_allow_html=True)
            #st.markdown('', unsafe_allow_html=True)
            
            # Match with jobs
            st.markdown(f'<h3><b>📊Recommended Jobs</b></h3>', unsafe_allow_html=True)
            
            job_matches = []
            
            for idx, row in jobs_df.iterrows():
                # Combine all relevant fields for skill matching
                # job_text = f"{row.get('job_title', '')} {row.get('description', '')} {row.get('requirements', '')} {row.get('skills', '')}"
                # job_skills = extract_job_skills(job_text)
                job_skills = row.get('skills', [])
                if isinstance(job_skills, str):
                    job_skills = ast.literal_eval(job_skills)

                # Ensure job_skills is always a list of lowercase strings
                # if isinstance(job_skills, str):
                # # Split by commas or spaces if needed
                #     job_skills = [s.strip().lower() for s in re.split(r'[,\s]+', job_skills) if s.strip()]
                # elif isinstance(job_skills, list):
                #     job_skills = [s.lower() for s in job_skills]
                # else:
                #     job_skills = []
                
                if job_skills:
                    result = calculate_match_score(extracted_skills, job_skills)
                    if isinstance(result, tuple):
                        score, matches = result
                    else:
                        score = result
                        matches = set()
                    
                    if score > 0:
                        job_matches.append({
                            'title': row.get('job_title', 'N/A'),
                            'company': row.get('company', 'N/A'),
                            'location': row.get('location_cleaned', 'N/A'),
                            'description': row.get('description', 'N/A'),
                            'score': score,
                            'matched_skills': set(matches),
                            'job_skills': set(job_skills)
                        })
            
            # Sort by score
            job_matches.sort(key=lambda x: x['score'], reverse=True)
            
            section_css = """
            <style>
            .section-title{
            margin:10px 0 5px;
            font-weight:700;
            font-size:18px;
            color:#334155;
            }
            </style>
            """
            st.markdown(section_css, unsafe_allow_html=True)

            # CSS for badges
            badge_css = """
            <style>
            .skill-badge {display:inline-block; margin:4px 6px 0 0; padding:6px 10px;
            border-radius:12px; font-weight:600; font-size:12px; white-space:nowrap;}
            .skill-badge.match {background:#16a34a; color:#fff;}        /* Matched */
            .skill-badge.job   {background:#334155; color:#e2e8f0;}     /* Other job skills */
            .skill-badge.user  {background:#0ea5e9; color:#fff;}        /* Optional: User-only */
            </style>
            """
            st.markdown(badge_css, unsafe_allow_html=True)

            if job_matches:
                st.markdown(f'Found {len(job_matches)} job matches', unsafe_allow_html=True)
                
                # Display top matches
                for i, job in enumerate(job_matches, 1):
                    matched = sorted({s.title() for s in job['matched_skills']})                               # Matched [A ∩ B]
                    job_only = sorted({s.title() for s in (job['job_skills'] - job['matched_skills'])})        # Job minus matched [B − A]
                    user_only = sorted({s.title() for s in (set(extracted_skills) - job['job_skills'])})       # Optional [A − B]

                    # Build badge HTML
                    matched_html  = "".join(f"<span class='skill-badge match'>{s}</span>" for s in matched)
                    job_only_html = "".join(f"<span class='skill-badge job'>{s}</span>" for s in job_only)
                    user_only_html = "".join(f"<span class='skill-badge user'>{s}</span>" for s in user_only)
                    st.markdown(f"""
                        <div class="job-card">
                            <div class="job-title">{i}.{job['title']}</div>
                            <div class="job-company">{job['company']} • {job['location']}</div>
                            <span class="match-score">{job['score']:.0f}% Match</span>
                            <div class="section-title">Matched skills</div>
                            <div class="matched-skills">
                                {matched_html}
                            </div>
                            <div class="section-title">Job skills</div>
                            <div class="matched-skills">
                                {job_only_html}
                            </div>
                            <div class="section-title">Your extra</div>
                            <div class="matched-skills">
                                {user_only_html}
                            </div>

                        </div>
                    """, unsafe_allow_html=True)
                

            else:
                st.info("No matching jobs found. Try adding more skills to your description!")

if __name__ == "__main__":

    main()

