import streamlit as st
import pandas as pd
import numpy as np
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

FILTERED_COURSES = None
SELECTED_COURSE = None

@st.cache_data
def clean_col_names(df, _columns):
    new = [c.lower().replace(' ', '_') for c in _columns]
    return new

@st.cache_data
def prepare_data(df):
    df.columns = clean_col_names(df, df.columns)
    df['skills'] = df['skills'].fillna('Missing')
    df['instructors'] = df['instructors'].fillna('Missing')

    def make_numeric(x):
        return np.nan if x == 'Missing' else float(x)

    df['course_rating'] = df['course_rating'].apply(make_numeric)
    df['course_rated_by'] = df['course_rated_by'].apply(make_numeric)
    df['percentage_of_new_career_starts'] = df['percentage_of_new_career_starts'].apply(make_numeric)
    df['percentage_of_pay_increase_or_promotion'] = df['percentage_of_pay_increase_or_promotion'].apply(make_numeric)

    def make_count_numeric(x):
        if 'k' in x:
            return float(x.replace('k', '')) * 1000
        elif 'm' in x:
            return float(x.replace('m', '')) * 1000000
        elif 'Missing' in x:
            return np.nan

    df['enrolled_student_count'] = df['enrolled_student_count'].apply(make_count_numeric)

    def find_time(x):
        l = x.split(' ')
        idx = next((i for i in range(len(l)) if l[i].isdigit()), 0)
        try:
            return f"{l[idx]} {l[idx+1]}"
        except:
            return l[idx]

    df['estimated_time_to_complete'] = df['estimated_time_to_complete'].apply(find_time)
    df['skills'] = df['skills'].apply(lambda x: x.split(','))

    return df

@st.cache_data
def load_data():
    df_overview = pd.read_csv('coursera-courses-overview.csv')
    df_individual = pd.read_csv('coursera-individual-courses.csv')
    df = pd.concat([df_overview, df_individual], axis=1)
    df = prepare_data(df)
    return df

@st.cache_data
def filter_data(dataframe, chosen_options, feature, id):
    selected_records = []
    for i in range(len(dataframe)):
        for op in chosen_options:
            if op in dataframe[feature][i]:
                selected_records.append(dataframe[id][i])
    return selected_records

def extract_keywords(df, feature):
    r = Rake()
    keyword_lists = []
    for i in range(len(df[feature])):
        descr = df[feature][i]
        r.extract_keywords_from_text(descr)
        key_words_dict_scores = r.get_word_degrees()
        keywords_string = " ".join(list(key_words_dict_scores.keys()))
        keyword_lists.append(keywords_string)
    return keyword_lists

def recommendations(df, input_course, cosine_sim, find_similar=True, how_many=5):
    recommended = []
    selected_course = df[df['course_name'] == input_course]
    idx = selected_course.index[0]

    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=not find_similar)
    top_sugg = list(score_series.iloc[1:how_many + 1].index)
    for i in top_sugg:
        qualified = df['course_name'].iloc[i]
        recommended.append(qualified)
    return recommended

def content_based_recommendations(df, input_course, courses):
    df = df[df['course_name'].isin(courses)].reset_index()
    df['descr_keywords'] = extract_keywords(df, 'description')
    count = CountVectorizer()
    count_matrix = count.fit_transform(df['descr_keywords'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    rec_courses_similar = recommendations(df, input_course, cosine_sim, True)
    temp_sim = df[df['course_name'].isin(rec_courses_similar)]
    rec_courses_dissimilar = recommendations(df, input_course, cosine_sim, False)
    temp_dissim = df[df['course_name'].isin(rec_courses_dissimilar)]

    st.write("Top 5 most similar courses")
    st.write(temp_sim)
    st.write("Top 5 most dissimilar courses")
    st.write(temp_dissim)

def prep_for_cbr(df):
    st.header("Content-based-Recommendation")
    st.sidebar.header("Filter on Preferences")
    st.write("This section analyses a filtered subset of courses based on skills.")
    skills_avail = [skill for sublist in df['skills'] for skill in sublist]
    skills_avail = list(set(skills_avail))
    skills_select = st.sidebar.multiselect("Select Skills", skills_avail)

    temp = filter_data(df, skills_select, 'skills', 'course_url')
    skill_filtered = df[df['course_url'].isin(temp)].reset_index()
    courses = skill_filtered['course_name']
    st.write("### Filtered courses based on skill preferences")
    st.write(skill_filtered)

    if len(courses) <= 2:
        st.write("*There should be at least 3 courses.*")
    input_course = st.sidebar.selectbox("Select Course", courses, key='courses')

    rec_radio = st.sidebar.radio("Recommend Similar Courses", ('no', 'yes'), index=0)
    if rec_radio == 'no':
        content_based_recommendations(df, input_course, courses)

def main():
    st.title("Course-recommendation-system")
    st.write("Made by Harshit Setia, Niharika bhatia, Akshat Tiwari")
    st.sidebar.title("Course selection")
    st.sidebar.header("Extra curricular")
    st.header("About the Project")
    st.write("Course recommendation System Suggest custom degree plans based on the student's career goals ")

    df = load_data()
    st.header("Coursera-courses")
    st.write("Data consists of 1000 instances and 14 features.")
    if st.sidebar.checkbox("Data Set view", key='disp_data'):
        st.write(df)

    prep_for_cbr(df)

if __name__ == "__main__":
    main()
