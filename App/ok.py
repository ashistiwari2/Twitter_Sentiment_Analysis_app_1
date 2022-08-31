#genism package
from gensim.summarization.summarizer import summarize
# NLTK Packages
import nltk
# nltk.download('stopwords')

# nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
#SPACY Packages
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
#Function for NLTK
def _create_frequency_table(text_string) -> dict:

    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable
    sent_tokenize(text_string)
def _score_sentences(sentences, freqTable) -> dict:
    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence

    return sentenceValue
def _find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))

    return average
def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def nltk_summarizer(text):
    freq_table=_create_frequency_table(text)
    sentences=sent_tokenize(text)
    sentence_score=_score_sentences(sentences,freq_table)
    threshold=_find_average_score(sentence_score)
    summary=_generate_summary(sentences,sentence_score,1.5*threshold)
    return summary

#Function for SPACY
def spacy_summarizer(docx):
    stopwords=list(STOP_WORDS) #buiding a list of stopword
    nlp=spacy.blank("en")
    nlp.add_pipe('sentencizer')
    docx1=nlp(docx)
    mytoken=[token.text for token in docx1]
    #build word frequency
    word_frequencies={}
    for word in docx1:
        if word.text not in word_frequencies.keys():
            word_frequencies[word.text]=1
        else:
            word_frequencies[word.text]+=1
    print(word_frequencies)
    #maximum wword frequencies
    maximum_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=(word_frequencies[word]/maximum_frequency)
    #frequency table
    print(word_frequencies)
    #sentence tokens
    sentence_list=[sentence for sentence in docx1.sents]
    #sentence score
    sentence_scores={}
    for sent in sentence_list:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' '))<30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent]=word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent]+=word_frequencies[word.text.lower()]

    from heapq import nlargest
    summarised_sentences=nlargest(7,sentence_scores,key=sentence_scores.get)
    for w in summarised_sentences:

        print(w.text)

    final_sentences=[w.text for w in summarised_sentences]
    summary=''.join(final_sentences)
    return summary
 elif choice=="Text summariser":
        st.title("Text Summarizer App")
        activities = ["Summarize Via Text"]
        choice = st.sidebar.selectbox("Select Activity", activities)

        if choice == 'Summarize Via Text':
            st.subheader("Summary using NLP")
            article_text = st.text_area("Enter Text Here", "Type here")
            # cleaning of input text
            # article_text = re.sub(r'\\[[0-9]*\\]', ' ', article_text)
            # article_text = re.sub('[^a-zA-Z.,]', ' ', article_text)
            # article_text = re.sub(r"\b[a-zA-Z]\b", '', article_text)
            # article_text = re.sub("[A-Z]\Z", '', article_text)
            # article_text = re.sub(r'\s+', ' ', article_text)

            summary_choice = st.selectbox("Summary Choice", ["NLTK", "SPACY","Genism"])
            if st.button("Summarize Via Text"):
                if summary_choice == 'NLTK':
                    summary_result = nltk_summarizer(article_text)
                elif summary_choice == 'SPACY':
                    summary_result = spacy_summarizer(article_text)
                elif summary_choice == 'Genism':
                    summary_result=summarize(article_text)

                st.write(summary_result