{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a829b57e-85e2-4938-a2ab-7935bdcca729",
   "metadata": {
    "panel-layout": {
     "height": 1018,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\CHARUSAT\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "2025-05-24 01:33:51.631 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run E:\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "# Top of file\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.cluster import KMeans\n",
    "from wordcloud import WordCloud\n",
    "import matplotlib.pyplot as plt\n",
    "import nltk\n",
    "nltk.download('stopwords')\n",
    "from nltk.corpus import stopwords\n",
    "# Title\n",
    "st.title(\"Research Keyword Analysis & Gap Detection\")\n",
    "\n",
    "# Upload Excel\n",
    "uploaded_file = st.file_uploader(r\"C:\\Users\\CHARUSAT\\Documents\\PhD\\scoupsdata.xlsx\", type=[\"xlsx\"])\n",
    "if uploaded_file:\n",
    "    df = pd.read_excel(uploaded_file)\n",
    "    df.columns = df.columns.str.strip().str.replace('ï»¿', '')\n",
    "    df['Combined'] = df['Title'].astype(str).str.lower() + ' ' + df['Abstract'].astype(str).str.lower()\n",
    "    \n",
    "    from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "    from sklearn.cluster import KMeans\n",
    "    from wordcloud import WordCloud\n",
    "    import matplotlib.pyplot as plt\n",
    "    import nltk\n",
    "    nltk.download('stopwords')\n",
    "    from nltk.corpus import stopwords\n",
    "\n",
    "    vectorizer = TfidfVectorizer(stop_words=stopwords.words(\"english\"), max_df=0.85, min_df=2)\n",
    "    X = vectorizer.fit_transform(df['Combined'].dropna())\n",
    "\n",
    "    kmeans = KMeans(n_clusters=5, random_state=42)\n",
    "    kmeans.fit(X)\n",
    "    df['Cluster'] = kmeans.labels_\n",
    "\n",
    "    tfidf_scores = X.mean(axis=0).A1\n",
    "    keywords = vectorizer.get_feature_names_out()\n",
    "    weights = dict(zip(keywords, (tfidf_scores / tfidf_scores.max() * 100).round(0).astype(int)))\n",
    "\n",
    "    st.subheader(\"Word Cloud of Keywords\")\n",
    "    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(weights)\n",
    "    import matplotlib.pyplot as plt\n",
    "    plt.figure(figsize=(14, 6))\n",
    "    plt.imshow(wordcloud, interpolation='bilinear')\n",
    "    plt.axis('off')\n",
    "    st.pyplot(plt)\n",
    "\n",
    "    # Suggested cluster\n",
    "    least_cluster = df['Cluster'].value_counts().idxmin()\n",
    "    st.subheader(f\"Suggested Research Gap - Cluster {least_cluster + 1}\")\n",
    "    for title in df[df['Cluster'] == least_cluster]['Title'].tolist():\n",
    "        st.markdown(f\"- {title}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1d6321b-496c-450c-9f5d-f1327a76896d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  },
  "panel-cell-order": [
   "a829b57e-85e2-4938-a2ab-7935bdcca729"
  ]
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
