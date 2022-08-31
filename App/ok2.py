
consumerKey ='yfYciQIR91EkR3eJ0ayLueNWQf'
consumerKey ='yfYciQIR91EkR3eJ0ayLueNWQf'
consumerSecret ='EhfHEq0TEjG6sGvqhCHybVUIaxe5hQt8rv8BYst30aiKtfrSQek'
accessToken ='1262567298931589120-KHsGvGNOfQQ2fWTnnSzM9kJZgJHKQ3D'
accessTokenSecret ='VrOycaetl1xmZw8CxRYvGlNf9Ohxd1PdhMUfLkT6W38fzw'
elif choice == "Monitor":
add_page_visited_details("Monitor", datetime.now())
st.subheader("Monitor App")
import plotly.express as px

with st.beta_expander("Page Metrics"):
    page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Pagename', 'Time_of_Visit'])
    st.dataframe(page_visited_details)

    pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
    c = alt.Chart(pg_count).mark_bar().encode(x='Pagename', y='Counts', color='Pagename')
    st.altair_chart(c, use_container_width=True)

    p = px.pie(pg_count, values='Counts', names='Pagename')
    st.plotly_chart(p, use_container_width=True)

with st.beta_expander('Emotion Classifier Metrics'):
    df_emotions = pd.DataFrame(view_all_prediction_details(),
                               columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
    st.dataframe(df_emotions)

    prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
    pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
    st.altair_chart(pc, use_container_width=True)
    add_page_visited_details("About", datetime.now())
