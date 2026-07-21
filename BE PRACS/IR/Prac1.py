from google.colab import files
# Upload Files
uploaded = files.upload()
# Read all Documents
documents = {}
for filename in uploaded.keys():
    with open(filename, "r") as file:
        documents[filename] = file.read()
print("Original Documents\n")
print(documents)

# ---------------- PREPROCESSING ----------------
stopwords = {
    "the","is","a","an","and","of","to","in",
    "on","for","with","has","have","been",
    "are","was","were","it","its","by","may"
}

processed_docs = {}
for name, text in documents.items():
    text = text.lower()
    words = text.split()
    cleaned = []
    for word in words:
        word = word.strip(".,!?()")
        if word not in stopwords:
            cleaned.append(word)

    processed_docs[name] = cleaned
print("\nProcessed Documents\n")
print(processed_docs)


# ---------------- MULTIPLE SEARCH ----------------
while True:
    query = input("\nEnter Search Query (type 'exit' to stop): ").lower()
    if query == "exit":
        break
    query_words = query.split()
    term_frequency = {}
    for name, words in processed_docs.items():
        count = 0
        for q in query_words:
            count += words.count(q)
        term_frequency[name] = count


    # -------- TERM FREQUENCY TABLE --------
    print("\nTerm Frequency Table")
    print("--------------------------------")
    print("Document\tFrequency")
    for doc, freq in term_frequency.items():
        print(doc, "\t", freq)

    # -------- RANK DOCUMENTS --------
    ranking = []
    for doc, freq in term_frequency.items():
        if freq > 0:
            ranking.append((doc, freq))
    ranking.sort(key=lambda x: x[1], reverse=True)
    print("\nRanked Documents")
    if len(ranking) == 0:
        print("No Relevant Documents Found")
    else:
        rank = 1
        for doc, score in ranking:
            print(f"Rank {rank}: {doc} --> Score = {score}")
            rank += 1
