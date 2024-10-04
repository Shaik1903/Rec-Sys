import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample user progress data over time
progress_data = {
    "User": ["Aiko", "Lucas", "Emma"],
    "Sessions": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
    "Fluency": [[55, 65, 70, 80, 85], [60, 62, 65, 70, 75], [50, 55, 60, 70, 78]],
    "Vocabulary": [[60, 65, 68, 72, 80], [58, 60, 62, 65, 70], [55, 58, 60, 65, 75]],
    "Grammar": [[58, 63, 70, 75, 80], [50, 55, 60, 65, 68], [52, 60, 65, 70, 72]],
    "Pronunciation": [[50, 60, 68, 75, 80], [55, 58, 60, 65, 72], [45, 50, 60, 68, 75]],
    "Common Mistakes": [[10, 8, 6, 4, 3], [12, 10, 8, 6, 4], [15, 12, 10, 8, 6]]
}

df = pd.DataFrame(progress_data)

# User selection (excluding "New User")
selected_user = st.selectbox("Select a User", df['User'])

# Filter data for selected user
user_data = df[df['User'] == selected_user].iloc[0]

# Display user information
st.write(f"**Selected User:** {user_data['User']}")
st.write(f"**Sessions:** {len(user_data['Sessions'])}")

# Button to generate output
if st.button("Generate Output"):
    st.write(f"Generated output for {selected_user}")

# Dashboard Section
st.header(f"Progress Dashboard for {selected_user}")

# 1. Fluency Progress (Line Chart)
st.subheader("Fluency Progress")
fluency_df = pd.DataFrame({
    'Session': user_data['Sessions'],
    'Fluency': user_data['Fluency']
})
st.line_chart(fluency_df.set_index('Session'))

# 2. Vocabulary Progress (Area Chart)
st.subheader("Vocabulary Progress")
vocab_df = pd.DataFrame({
    'Session': user_data['Sessions'],
    'Vocabulary': user_data['Vocabulary']
})
st.area_chart(vocab_df.set_index('Session'))

# 3. Grammar Progress (Bar Chart)
st.subheader("Grammar Progress")
grammar_df = pd.DataFrame({
    'Session': user_data['Sessions'],
    'Grammar': user_data['Grammar']
})
st.bar_chart(grammar_df.set_index('Session'))

# 4. Pronunciation Progress (Scatter Plot)
st.subheader("Pronunciation Progress")
pronunciation_df = pd.DataFrame({
    'Session': user_data['Sessions'],
    'Pronunciation': user_data['Pronunciation']
})

# Matplotlib scatter plot for Pronunciation
fig, ax = plt.subplots()
ax.scatter(pronunciation_df['Session'], pronunciation_df['Pronunciation'], color='green')
ax.set_xlabel('Session')
ax.set_ylabel('Pronunciation Score')
ax.set_title('Pronunciation Progress')
st.pyplot(fig)

# 5. Common Mistakes Reduction (Line Chart)
st.subheader("Common Mistakes Reduction")
mistakes_df = pd.DataFrame({
    'Session': user_data['Sessions'],
    'Common Mistakes': user_data['Common Mistakes']
})
st.line_chart(mistakes_df.set_index('Session'))

# Summary Statistics
st.subheader("Summary of Progress")
st.write(f"User {selected_user} has completed {len(user_data['Sessions'])} sessions.")
st.write(f"Fluency improvement: {user_data['Fluency'][0]} → {user_data['Fluency'][-1]}")
st.write(f"Vocabulary improvement: {user_data['Vocabulary'][0]} → {user_data['Vocabulary'][-1]}")
st.write(f"Grammar improvement: {user_data['Grammar'][0]} → {user_data['Grammar'][-1]}")
st.write(f"Pronunciation improvement: {user_data['Pronunciation'][0]} → {user_data['Pronunciation'][-1]}")
st.write(f"Common mistakes reduced from {user_data['Common Mistakes'][0]} to {user_data['Common Mistakes'][-1]}")
