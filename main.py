import re
import nltk
import spacy
import gensim
import joblib
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from spacy import displacy
from sklearn import metrics
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("en_core_web_lg")
nltk.download(['stopwords', 'wordnet', 'punkt'], quiet=True)

# Log file configuration
logging.basicConfig(level=logging.INFO,
                    filename='Outputs/log.log',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='[%(asctime)s] - [%(levelname)s] - [%(funcName)s] --> %(message)s')


class Visualizer:
    def __init__(self, df):
        self.df = df

    def plot_category_distribution(self):
        try:
            count = self.df['Category'].value_counts()
            labels = [f'{label} ({count[label]})' for label in count.index]

            plt.figure(figsize=(12, 12))
            plt.pie(count, labels=labels, autopct='%1.2f%%')
            plt.axis('equal')
            plt.savefig('Outputs/category_distribution.jpg')
            plt.close()

            logging.info('Category distribution plotted successfully.')
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)

    def plot_word_frequency(self):
        try:
            categories = self.df['Category'].unique()
            df_categories = [self.df[self.df['Category'] == category] for category in categories]

            fig = plt.figure(figsize=(32, 64))

            # Plot 10 most used words for each category
            for i, category in enumerate(categories):
                wf = self._word_frequency(df_categories[i])

                fig.add_subplot(12, 2, i + 1).set_title(category)
                plt.bar(wf['Word'], wf['Frequency'])
                plt.ylim(0, 3500)

            plt.savefig('Outputs/Word Frequency.jpg')
            plt.close()

            logging.info('Word frequency plotted successfully.')
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)

    def plot_skills_distribution(self, skills_counter: pd.Series, job_category: str = "HR"):
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(skills_counter.index, skills_counter.values)
            plt.xlabel('Skills')
            plt.xticks(rotation=90, fontsize=5)
            plt.ylabel('Count')
            plt.title(f'{job_category} Distribution of Skills')

            plt.savefig(f'Outputs/{job_category} Distribution of Skills.jpg')
            plt.close()

            logging.info('Skills distribution plotted successfully.')
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)

    def plot_most_used_category_words(self, job_category: str = "HR"):
        try:
            text = ""
            for i in self.df[self.df["Category"] == job_category]["Clean_Resume"].values:
                text += i + " "

            plt.figure(figsize=(8, 8))

            x, y = np.ogrid[:300, :300]

            mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
            mask = 255 * mask.astype(int)

            wc = WordCloud(
                width=800,
                height=800,
                background_color="white",
                min_font_size=6,
                repeat=True,
                mask=mask,
            )
            wc.generate(text)

            plt.axis("off")
            plt.imshow(wc, interpolation="bilinear")
            plt.title(f"Most Used Words in {job_category} Resume", fontsize=20)
            plt.savefig(f'Outputs/Most Used Words in {job_category} Resume.jpg')

            logging.info('Most used category words plotted successfully.')
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)

    def plot_displacy_entities(self):
        try:
            # Entity Recognition
            sent = nlp(self.df["Resume_str"].iloc[0])

            # Render the visualization to a file
            html = displacy.render(sent, style="ent")

            # Write the file
            with open("Outputs/displacy.html", "w", encoding="utf-8") as f:
                f.write(html)
            logging.info('Displacy entities plotted successfully.')
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)

    def plot_displacy_dependency(self):
        try:
            # Dependency Parsing
            sent = nlp(self.df["Resume_str"].iloc[0])

            # Render the dependency tree visualization to a file
            html = displacy.render(sent[0:10], style="dep", options={"distance": 90})

            # Write the file
            with open("Outputs/displacy_dep.html", "w", encoding="utf-8") as f:
                f.write(html)
            logging.info('Displacy dependency plotted successfully.')
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)

    @staticmethod
    def _word_frequency(df):
        count = df['Resume'].str.split(expand=True).stack().value_counts().reset_index()
        count.columns = ['Word', 'Frequency']
        return count.head(10)


class Preprocessor:
    def __init__(self, dataset_path: str):
        self.df = pd.read_csv(dataset_path)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def run(self):
        self._preprocess_data()

        visualizer = Visualizer(df=self.df)
        visualizer.plot_word_frequency()
        visualizer.plot_category_distribution()

        return self.df

    def _preprocess_data(self):
        self.df['Resume'] = self.df['Resume_str'].apply(self._preprocess_text)

    def _preprocess_text(self, txt: str):
        txt = txt.lower()
        txt = re.sub('[^a-zA-Z]', ' ', txt)
        txt = word_tokenize(txt)
        txt = [w for w in txt if w not in self.stop_words]
        txt = [self.stemmer.stem(w) for w in txt]
        return ' '.join(txt)


class SpacyModel:
    def __init__(self, df):
        self.df = df
        self.ruler = nlp.add_pipe("entity_ruler")
        skill_pattern_path = "jz_skill_patterns.jsonl"
        self.ruler.from_disk(skill_pattern_path)

    def run(self):
        self._preprocess_data()
        self._clean_data()
        skills_counter = self._process_job_category_skills()

        visualizer = Visualizer(df=self.df)
        visualizer.plot_skills_distribution(skills_counter=skills_counter)
        visualizer.plot_most_used_category_words()
        visualizer.plot_displacy_entities()
        visualizer.plot_displacy_dependency()

    def _preprocess_data(self):
        self.df["Clean_Resume"] = self._clean_resume_text()
        self.df["skills"] = self.df["Clean_Resume"].str.lower().apply(self._get_skills)
        self.df["skills"] = self.df["skills"].apply(self._unique_skills)

    def _clean_data(self):
        self.df['clean'] = self.df['Resume'].apply(self._remove_stop_words).astype(str)

    @staticmethod
    def _remove_stop_words(text):
        stop_words = stopwords.words('english')

        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
                result.append(token)

        return result

    def _process_job_category_skills(self, job_category: str = "HR"):
        if job_category != "ALL":
            skills = self.df[self.df["Category"] == job_category]["skills"].apply(pd.Series).stack().tolist()
        else:
            skills = self.df["skills"].apply(pd.Series).stack().tolist()

        return pd.Series(skills).value_counts()

    def _clean_resume_text(self):
        clean = []
        for i in range(self.df.shape[0]):
            review = re.sub(
                '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
                " ",
                self.df["Resume_str"].iloc[i],
            )
            review = review.lower()
            review = review.split()
            lm = WordNetLemmatizer()
            review = [
                lm.lemmatize(word)
                for word in review
                if word not in set(stopwords.words("english"))
            ]
            review = " ".join(review)
            clean.append(review)
        return clean

    @staticmethod
    def _get_skills(text):
        doc = nlp(text)
        subset = []
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                subset.append(ent.text)
        return subset

    @staticmethod
    def _unique_skills(x):
        return list(set(x))


class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.label_encoder = LabelEncoder()
        self.xgb = xgb.XGBClassifier()
        self.rfc = RandomForestClassifier(random_state=42)

    def run(self):
        self._split_data()
        train_vectorizer, test_vectorizer = self._vectorize_data()
        grid_search = self._find_optimal_values(train_vectorizer=train_vectorizer)

        self._train_model_rfc(grid_search=grid_search, train_vectorizer=train_vectorizer)
        self._predict_rfc(train_vectorizer=train_vectorizer, test_vectorizer=test_vectorizer)

        self._train_model_xgb(grid_search=grid_search, train_vectorizer=train_vectorizer)
        self._predict_xgb(train_vectorizer=train_vectorizer, test_vectorizer=test_vectorizer)

    def _split_data(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.df['clean'], self.df['Category'],
                                                                                test_size=0.2, shuffle=True)

        # Encode target variable
        self.Y_train = self.label_encoder.fit_transform(self.Y_train)
        self.Y_test = self.label_encoder.transform(self.Y_test)

    def _vectorize_data(self):
        vectorizer = CountVectorizer()
        train_vectorizer = vectorizer.fit_transform(self.X_train).astype(float)
        test_vectorizer = vectorizer.transform(self.X_test).astype(float)
        return train_vectorizer, test_vectorizer

    def _find_optimal_values(self, train_vectorizer):
        param_grid = {
            'n_estimators': [200, 500, 800],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [8, 9, 10, 11, 13],
            'criterion': ['gini', 'entropy']
        }

        grid = GridSearchCV(cv=5,
                            verbose=1,
                            scoring='accuracy',
                            estimator=self.rfc,
                            param_grid=param_grid,
                            return_train_score=False)
        grid_search = grid.fit(train_vectorizer, self.Y_train)
        logging.info(grid_search.best_params_)

        return grid_search

    def _train_model_rfc(self, grid_search, train_vectorizer):
        self.rfc = RandomForestClassifier(random_state=42,
                                          max_features=grid_search.best_params_['max_features'],
                                          max_depth=grid_search.best_params_['max_depth'],
                                          criterion=grid_search.best_params_['criterion'],
                                          n_estimators=grid_search.best_params_['n_estimators'])
        self.rfc.fit(train_vectorizer, self.Y_train)

    def _predict_rfc(self, train_vectorizer, test_vectorizer):
        prediction = self.rfc.predict(test_vectorizer)

        logging.info("RFC Training Score: {:.2f}".format(self.rfc.score(train_vectorizer, self.Y_train)))
        logging.info("RFC Test Score: {:.2f}".format(self.rfc.score(test_vectorizer, self.Y_test)))

        logging.info("model report: %s: \n %s\n" % (self.rfc, metrics.classification_report(self.Y_test, prediction)))

        # Save the trained model
        filename = 'Outputs/rfc_model.sav'
        joblib.dump(self.rfc, filename)
        logging.info('Trained model saved successfully.')

    def _train_model_xgb(self, grid_search, train_vectorizer):
        self.xgb = xgb.XGBClassifier(gamma=0,
                                     verbosity=0,
                                     reg_lambda=1,
                                     random_state=42,
                                     max_depth=grid_search.best_params_['max_depth'],
                                     n_estimators=grid_search.best_params_['n_estimators'])
        self.xgb.fit(train_vectorizer, self.Y_train)

    def _predict_xgb(self, train_vectorizer, test_vectorizer):
        prediction = self.xgb.predict(test_vectorizer)
        predicted_categories = self.label_encoder.inverse_transform(prediction)

        logging.info("XGB Training Score: {:.2f}".format(self.xgb.score(train_vectorizer, self.Y_train)))
        logging.info("XGB Test Score: {:.2f}".format(self.xgb.score(test_vectorizer, self.Y_test)))

        logging.info("model report: %s: \n %s\n" % (self.xgb, metrics.classification_report(self.Y_test, prediction)))

        # Save the trained model
        filename = 'Outputs/xgb_model.sav'
        joblib.dump(self.xgb, filename)
        logging.info('Trained model saved successfully.')


def main():
    logging.info("Execution started..")

    # First Stage
    preprocessor = Preprocessor(dataset_path='Resume/Resume.csv')
    df = preprocessor.run()

    # Second Stage
    spacy_model = SpacyModel(df=df)
    spacy_model.run()

    # Third Stage
    model = ModelTrainer(df=df)
    model.run()


if __name__ == '__main__':
    main()
