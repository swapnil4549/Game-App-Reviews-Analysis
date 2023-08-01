import streamlit as st
import pandas as pd
import pickle
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
snow_stemmer = SnowballStemmer("english")
import re
# from sklearn.linear_model import LogisticRegression 
import base64
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )
# add_bg_from_local('photo-game1.jpg')  

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(snow_stemmer.stem(i))

    return " ".join(y)

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F" # emoticons
                           u"\U0001F300-\U0001F5FF" # symbols & pictographs
                           u"\U0001F680-\U0001F6FF" # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
tfidfimmersion = pickle.load(open('vectorizer1.pkl','rb'))
tfidfEffectiveness = pickle.load(open('vectorizer2.pkl','rb'))
tfidfSatisfaction = pickle.load(open('vectorizer3.pkl','rb'))
tfidfLearnability = pickle.load(open('vectorizer4.pkl','rb'))
tfidfMotivation = pickle.load(open('vectorizer5.pkl','rb'))
tfidfEmotion = pickle.load(open('vectorizer6.pkl','rb'))
modelnew = pickle.load(open('modelnew.pkl','rb'))
modelforImmersion = pickle.load(open('modelforImmersion.pkl','rb'))
modelforEffectiveness = pickle.load(open('modelforEffectiveness.pkl','rb'))
modelforSatisfaction1 = pickle.load(open('modelforSatisfaction1.pkl','rb'))
modelforLearnability = pickle.load(open('modelforLearnability.pkl','rb'))
modelforMotivation = pickle.load(open('modelforMotivation.pkl','rb'))
modelforEmotion = pickle.load(open('modelforEmotion.pkl','rb'))

st.title("Game Playability Value Classifier based on review")

input_sms = st.text_area("Enter the Review")

if st.button('Make Prediction'):

    # 1. preprocess
    new_input_sms = remove_emoji(input_sms)
    transformed_sms = transform_text(new_input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    vector_input1 = tfidfimmersion.transform([transformed_sms])
    vector_input2 = tfidfEffectiveness.transform([transformed_sms])
    vector_input3 = tfidfSatisfaction.transform([transformed_sms])
    vector_input4 = tfidfLearnability.transform([transformed_sms])
    vector_input5 = tfidfMotivation.transform([transformed_sms])
    vector_input6 = tfidfEmotion.transform([transformed_sms])
    # st.header(vector_input)
    # 3. predict
    # lrc.fit(x,vector_input)
    result = modelnew.predict(vector_input)[0]
    resultforImmersion = modelforImmersion.predict(vector_input1)[0]
    resultforEffectiveness = modelforEffectiveness.predict(vector_input2)[0]
    resultforSatisfaction = modelforSatisfaction1.predict(vector_input3)[0]
    resultforLernability = modelforLearnability.predict(vector_input4)[0]
    resultforMotivation = modelforMotivation.predict(vector_input5)[0]
    resultforEmotion = modelforEmotion.predict(vector_input6)[0]
    # st.header(result)
    # 4. Display
    if result == 1:
        st.subheader(":red[Socialism] Violated Review")
    if resultforImmersion == 1:
        st.subheader(":red[Immersion] Violated Review")
    if resultforEffectiveness == 1:
        st.subheader(":red[Effectiveness] Violated Review")
    if resultforSatisfaction == 1:
        st.subheader(":red[Satisfaction] Violated Review")
    if resultforLernability == 1:
        st.subheader(":red[Learnability] Violated Review")
    if resultforMotivation == 1:
        st.subheader(":red[Motivation] Violated Review")
    if resultforEmotion == 1:
        st.subheader(":red[Emotion] Violated Review")
        

st.title('Playability Model by Sánchez el al.')

    # Sample data in a dictionary
data = {
    'Atribute': ['Satisfaction', 'Learnability', 'Effectiveness', 'Immersion', 'Motivation','Emotion','Socialisam'],
    'Properties': ['Fun, Disappointment, Attractiveness',
    'Game Knowledge, Skill, Difficulty, Frustration,Speed, Discovery',
    'Completion, Structuring',
    'Conscious Awareness, Absorption, Realism,Dexterity, Socio-Cultural Proximity',
    'Encouragement, Curiosity, Self-improvement,Diversity',
    'Reaction, Conduct, Sensory Appeal',
    'Social Perception, Group Awareness, Personal Implication, Sharing, Communication,Interaction'],
}

    # Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

    # Display the table using st.table()
st.table(df)