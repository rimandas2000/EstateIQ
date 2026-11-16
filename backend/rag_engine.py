"""Small RAG index. Uses ChromaDB when installed; TF-IDF fallback keeps demo runnable."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

class PropertyRAG:
    def __init__(self):
        self.docs=[]; self.collection=None; self.vectorizer=None; self.matrix=None
        self._load_docs()
        try:
            import chromadb
            self.client=chromadb.PersistentClient(path=str(ROOT/'data'/'chroma'))
            self.collection=self.client.get_or_create_collection('estateiq_knowledge')
            if self.collection.count()==0 and self.docs:
                self.collection.add(ids=[str(i) for i in range(len(self.docs))],documents=self.docs)
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2))
            self.matrix=self.vectorizer.fit_transform(self.docs) if self.docs else None
    def _load_docs(self):
        path=ROOT/'data'/'properties.csv'
        if path.exists():
            df=pd.read_csv(path)
            self.docs=[f"{r.city}, {r.locality}: {r.property_type}, {r.bedrooms} BHK, {r.area_sqft} sq ft, asking sample ₹{r.price_inr:,.0f}." for r in df.itertuples()]
        self.docs += [
          'Property valuation depends on locality, carpet or built-up area, building age, floor, amenities, furnishing and connectivity.',
          'Compare multiple recent transactions and active listings before making a purchase decision. Asking prices are not the same as final transaction prices.',
          'A price per square foot comparison is meaningful only when property type, area definition, age and locality are comparable.',
          'Rental yield can be estimated as annual rent divided by property purchase price, multiplied by 100. Consider maintenance, vacancy and taxes.'
        ]
    def retrieve(self, query, k=5):
        if self.collection is not None:
            try:
                res=self.collection.query(query_texts=[query],n_results=min(k,max(1,self.collection.count())))
                return res.get('documents',[[]])[0]
            except Exception: pass
        if self.vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer=TfidfVectorizer(stop_words='english'); self.matrix=self.vectorizer.fit_transform(self.docs)
        q=self.vectorizer.transform([query]); scores=(self.matrix@q.T).toarray().ravel()
        return [self.docs[i] for i in scores.argsort()[::-1][:k] if scores[i]>0]

rag=PropertyRAG()
