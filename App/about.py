import streamlit as st
from PIL import Image
import pandas as pd
def about():
    st.subheader("About")
    st.write("Created by Ashis and Akshat ")
    image = Image.open(
        'App/Ashis_recent_passport_size_photo.jpeg')
    st.write('------------------------------------------------------------------------------------------')
    st.image(image, caption="Ashis Tiwari's image", width=200, channels="RGB", output_format="auto")
    st.write('------------------------------------------------------------------------------------------')
    st.write(
        "Ashis Tiwari is thrilling student with adventuros nature. He is in final year of college and student of Sikkim"
        "  manipal institute of technology. He did his schooling from Pinegrove School,Dharampur ")
    st.write('------------------------------------------------------------------------------------------')
    image1 = Image.open(
        "App/WhatsApp Image 2022-04-02 at 1.21.13 PM.jpeg")
    st.image(image1, caption="Akshat Kedia's image", width=200, channels="RGB", output_format="auto")

    st.write('------------------------------------------------------------------------------------------')
    st.write("Akshat kedia is doing his btech from SMIT and in Final year")
    st.write('------------------------------------------------------------------------------------------')
    st.header("Our Guide")
    st.write('------------------------------------------------------------------------------------------')
    image3 = Image.open('App/Biraj.jpg')
    st.image(image3, caption="Biraj Upadhyaya |Assistant Professor |Department of Computer Science & Engineering"
             , width=200)
    st.write('------------------------------------------------------------------------------------------')
    st.header('CURRENT ACADEMIC ROLE & RESPONSIBILITIES')
    st.write(
        'Biraj Upadhyaya is an Assistant Professor I in the Department of Computer Science & Engineering. In addition to this, he is responsible for the followings:')
    st.write('1.Semester Coordinator')
    st.write('2.Mini Project Coordinator')
    st.header("SUBJECTS CURRENTLY TEACHING")
    subjectlist = [['Unix Internals and Shell Programming', 'CS1605', 'Sixth'],
                   ['High Performance Computing', 'CS1645', 'Sixth'],
                   ['Optimization Techniques', 'CS1736', 'Seventh'],
                   ['Soft Computing', '	CS 1702', '	Seventh'],
                   ['System Simulation and Modeling', 'CS1638', 'Sixth'],
                   ['Software Engineering', 'CS1504', '	Fifth'],
                   ['Computer Organization and Architecture', 'CS1306', 'Third'],
                   ['Design and Analysis of Algorithms', 'CS 1405', '	II/IV'],
                   ['Discrete Structures', 'CS 1507', '	III/V'],
                   ['Data Structure', 'CS 1302', 'III'],
                   ['Fundamentals of Web Technologies', 'CS 1436/CS 1642 (Open Electives)', 'IV'],
                   ['Parallel Programming Lab', 'CS1666', 'Sixth Semester']]
    cols = ['Subject', 'Subject code', 'Semester']

    df1 = pd.DataFrame(subjectlist, columns=cols)
    st.dataframe(df1, 600, 600)
    st.header("ACADEMIC QUALIFICATIONS")
    degree = [['Master of Technology', 'Computer Science & Engineering', 'Sikkim Manipal Institute of Technology',
               '2015'],
              ['Bachelor of Engineering', 'Computer Engineering', 'Institute Of Engineering, Tribhuvan University',
               '2011']]
    cols1 = ['Degree', 'Specialisation', 'Institute', 'Year of passing']
    df2 = pd.DataFrame(degree, columns=cols1)
    st.table(df2)
    st.header("EXPERIENCE")
    cols2 = ['Institution / Organisation', 'Designation', 'Role', 'Tenure']
    exp = [['Sikkim Manipal Institute of Technology', '	Assistant Professor II', '  ', 'Aug 2015 - till date']]
    df3 = pd.DataFrame(exp, columns=cols2)
    st.table(df3)
    st.header("AREAS OF INTEREST, EXPERTISE AND RESEARCH")
    st.write("AREA OF INTEREST->                Operating System, Soft Computing, High performance computing")
    st.write("AREA OF EXPERTISE->              Soft Computing, High Performance Computing")
    st.write("AREA OF RESEARCH->               Natural Language Processing")
    st.header("PUBLICATION")
    st.subheader("A Survey on Formulation of Faulty Class Detection System for Object Oriented Software")
    st.write("January 31, 2015 |     Software Engineering   |  Biraj Upadhyaya")
    st.write("International Journal of Innovative Science, Engineering & Technology, Vol. 2 Issue 2, February 2015")
    st.subheader("A Survey on various stemming techniques for Hindi and Nepali Language")
    st.write("Biraj Upadhyaya  |    B. Upadhyaya  |   S. Gurung  |  K. Sharma")
    st.write(
        "A Survey on various stemming techniques for Hindi and Nepali Language. 4th International Conference on Communication,")
    st.write("Devices and Networking (ICCDN 2020). International/ Scopus Indexed")
    st.subheader("A Survey on various stemming techniques for Hindi and Nepali Language")
    st.write("Biraj Upadhyaya |     B. Upadhyaya |    S. Gurung |   K. Sharma")
    st.write(
        "A Survey on various stemming techniques for Hindi and Nepali Language. 4th International Conference on Communication,")
    st.write("Devices and Networking (ICCDN 2020). International/ Scopus Indexed Yet to get published")

    st.write('------------------------------------------------------------------------------------------')
    st.write('------------------------------------------------------------------------------------------')
