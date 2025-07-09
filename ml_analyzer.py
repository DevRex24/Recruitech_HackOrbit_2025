import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging
import pickle
import os

class ResumeAnalyzer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.count_vectorizer = CountVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.job_classification_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.score_prediction_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        self.ats_compatibility_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        self.models_trained = False
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                          'typescript', 'kotlin', 'swift', 'scala', 'r', 'matlab', 'sql'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express',
                              'django', 'flask', 'bootstrap', 'jquery', 'webpack'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
                           'matplotlib', 'seaborn', 'jupyter', 'tableau', 'powerbi'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle'],
            'tools': ['git', 'jira', 'confluence', 'slack', 'trello', 'figma', 'photoshop']
        }
        self.career_fields = {
            'Software Engineering': {
                'keywords': ['software', 'development', 'programming', 'coding', 'algorithm', 'debugging', 
                           'application', 'system', 'backend', 'frontend', 'fullstack', 'engineer'],
                'skills': ['programming', 'databases', 'tools'],
                'weight': 1.0
            },
            'Data Science & Analytics': {
                'keywords': ['data', 'analytics', 'machine learning', 'statistics', 'analysis', 'modeling',
                           'visualization', 'insights', 'research', 'scientist', 'analyst'],
                'skills': ['data_science', 'programming', 'databases'],
                'weight': 1.0
            },
            'Frontend Development': {
                'keywords': ['frontend', 'ui', 'ux', 'interface', 'responsive', 'design', 'user experience',
                           'web design', 'interactive', 'visual'],
                'skills': ['web_development', 'programming', 'tools'],
                'weight': 1.0
            },
            'DevOps & Cloud Engineering': {
                'keywords': ['devops', 'infrastructure', 'deployment', 'automation', 'monitoring', 'cicd',
                           'containerization', 'orchestration', 'reliability', 'operations'],
                'skills': ['cloud', 'programming', 'tools'],
                'weight': 1.0
            },
            'Product Management': {
                'keywords': ['product', 'strategy', 'roadmap', 'requirements', 'stakeholder', 'agile',
                           'scrum', 'market', 'business', 'management', 'planning', 'coordination'],
                'skills': ['tools'],
                'weight': 0.8
            },
            'Digital Marketing': {
                'keywords': ['marketing', 'digital', 'social media', 'campaigns', 'seo', 'sem', 'content',
                           'brand', 'advertising', 'analytics', 'engagement', 'growth'],
                'skills': ['tools'],
                'weight': 0.7
            },
            'Backend Development': {
                'keywords': ['backend', 'server', 'api', 'microservices', 'architecture', 'scalability',
                           'performance', 'integration', 'services'],
                'skills': ['programming', 'databases', 'cloud'],
                'weight': 1.0
            },
            'Full Stack Development': {
                'keywords': ['fullstack', 'full stack', 'end-to-end', 'complete', 'comprehensive'],
                'skills': ['programming', 'web_development', 'databases'],
                'weight': 1.0
            },
            'Database Administration': {
                'keywords': ['database', 'dba', 'administration', 'optimization', 'performance tuning',
                           'backup', 'recovery', 'maintenance'],
                'skills': ['databases', 'programming'],
                'weight': 0.9
            },
            'Quality Assurance': {
                'keywords': ['qa', 'quality', 'testing', 'automation', 'test cases', 'bug', 'validation',
                           'verification', 'assurance'],
                'skills': ['programming', 'tools'],
                'weight': 0.8
            }
        }
        self._initialize_models()
    def _initialize_models(self):
        try:
            if self._load_models():
                self.logger.info("Using pre-trained models")
                return
            if os.path.exists('resume_dataset.csv'):
                training_data = self._load_dataset_from_csv('resume_dataset.csv')
                self.logger.info("Loaded training data from CSV file")
            else:
                training_data = self._generate_training_data()
                self.logger.info("Generated synthetic training data")
            self._train_models(training_data)
            self.models_trained = True
            self.logger.info("ML models successfully trained")
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            self.models_trained = False
    def _load_dataset_from_csv(self, csv_file):
        try:
            df = pd.read_csv(csv_file)
            features = []
            labels = []
            match_scores = []
            ats_scores = []
            for _, row in df.iterrows():
                feature_vector = self._extract_features(row['resume_text'], row['job_description'])
                features.append(feature_vector)
                labels.append(row['job_category'])
                match_scores.append(row['match_score'])
                ats_scores.append(row['ats_score'])
            self.logger.info(f"Loaded {len(features)} samples from CSV dataset")
            return {
                'features': np.array(features),
                'labels': labels,
                'match_scores': np.array(match_scores),
                'ats_scores': np.array(ats_scores)
            }
        except Exception as e:
            self.logger.error(f"Error loading CSV dataset: {str(e)}")
            return self._generate_training_data()
    def _generate_training_data(self):
        sample_resumes = {
            'Software Engineering': [
                "Software engineer with 5 years experience in Python, Java, and JavaScript. Developed web applications using React and Node.js. Strong background in algorithms and data structures.",
                "Full-stack developer skilled in Python, Django, React, and PostgreSQL. Built scalable web applications with microservices architecture. Experience with AWS and Docker.",
                "Backend developer with expertise in Java, Spring Boot, and MySQL. Developed RESTful APIs and microservices. Strong understanding of system design and architecture."
            ],
            'Data Science & Analytics': [
                "Data scientist with expertise in Python, R, and machine learning. Experience with pandas, scikit-learn, and TensorFlow. Built predictive models for business insights.",
                "Analytics professional skilled in SQL, Python, and Tableau. Experience with statistical analysis and data visualization. Strong background in business intelligence.",
                "ML engineer with experience in deep learning, computer vision, and NLP. Proficient in PyTorch, TensorFlow, and MLOps practices."
            ],
            'Frontend Development': [
                "Frontend developer with 3 years experience in React, Vue.js, and Angular. Strong skills in HTML, CSS, and JavaScript. Experience with responsive design and UX principles.",
                "UI/UX developer proficient in React, TypeScript, and modern CSS frameworks. Experience with design systems and user-centered design methodologies.",
                "Web developer with expertise in JavaScript, HTML5, CSS3, and modern frameworks. Experience with performance optimization and accessibility."
            ],
            'DevOps & Cloud Engineering': [
                "DevOps engineer with experience in AWS, Docker, and Kubernetes. Strong background in CI/CD pipelines and infrastructure automation using Terraform.",
                "Cloud architect skilled in Azure, containerization, and microservices. Experience with monitoring, logging, and site reliability engineering.",
                "Infrastructure engineer with expertise in AWS, Jenkins, and automation tools. Experience with scalable system design and deployment strategies."
            ],
            'Product Management': [
                "Product manager with 4 years experience in agile methodologies and product strategy. Strong background in user research and market analysis.",
                "Senior product manager skilled in roadmap planning, stakeholder management, and cross-functional team leadership. Experience with data-driven decision making.",
                "Product owner with experience in scrum, user story creation, and product lifecycle management. Strong analytical and communication skills."
            ]
        }
        sample_jobs = {
            'Software Engineering': [
                "Software engineer position requiring Python, Java, and web development skills. Experience with databases and cloud platforms preferred.",
                "Full-stack developer role focusing on React, Node.js, and database design. Strong problem-solving skills and agile experience required.",
                "Backend developer position requiring API development, microservices, and cloud deployment experience."
            ],
            'Data Science & Analytics': [
                "Data scientist role requiring Python, machine learning, and statistical analysis skills. Experience with big data tools preferred.",
                "Analytics position focusing on SQL, Python, and business intelligence. Strong communication skills for presenting insights to stakeholders.",
                "ML engineer role requiring deep learning, model deployment, and MLOps experience."
            ],
            'Frontend Development': [
                "Frontend developer position requiring React, JavaScript, and responsive design skills. UX/UI experience preferred.",
                "UI developer role focusing on modern JavaScript frameworks and design systems. Strong attention to detail required.",
                "Web developer position requiring HTML, CSS, JavaScript, and performance optimization skills."
            ],
            'DevOps & Cloud Engineering': [
                "DevOps engineer role requiring AWS, Docker, and CI/CD pipeline experience. Infrastructure as code knowledge preferred.",
                "Cloud engineer position focusing on containerization, monitoring, and scalable system design.",
                "Site reliability engineer role requiring automation, monitoring, and incident response experience."
            ],
            'Product Management': [
                "Product manager position requiring agile experience, user research, and roadmap planning skills.",
                "Senior product manager role focusing on strategy, stakeholder management, and cross-functional leadership.",
                "Product owner position requiring scrum experience, user story creation, and analytical skills."
            ]
        }
        training_samples = []
        labels = []
        match_scores = []
        ats_scores = []
        for job_category, resumes in sample_resumes.items():
            for resume in resumes:
                for job_cat, jobs in sample_jobs.items():
                    for job in jobs:
                        features = self._extract_features(resume, job)
                        if job_category == job_cat:
                            match_score = np.random.normal(85, 10)
                            ats_score = np.random.normal(80, 8)
                        elif self._similar_categories(job_category, job_cat):
                            match_score = np.random.normal(65, 15)
                            ats_score = np.random.normal(75, 10)
                        else:
                            match_score = np.random.normal(35, 20)
                            ats_score = np.random.normal(60, 15)
                        match_score = np.clip(match_score, 0, 100)
                        ats_score = np.clip(ats_score, 0, 100)
                        training_samples.append(features)
                        labels.append(job_category)
                        match_scores.append(match_score)
                        ats_scores.append(ats_score)
        return {
            'features': np.array(training_samples),
            'labels': labels,
            'match_scores': np.array(match_scores),
            'ats_scores': np.array(ats_scores)
        }
    def _similar_categories(self, cat1, cat2):
        similar_pairs = [
            ('Software Engineering', 'Backend Development'),
            ('Software Engineering', 'Frontend Development'),
            ('Backend Development', 'DevOps & Cloud Engineering'),
            ('Frontend Development', 'Software Engineering'),
            ('Data Science & Analytics', 'Software Engineering')
        ]
        return (cat1, cat2) in similar_pairs or (cat2, cat1) in similar_pairs
    def _extract_features(self, resume_text, job_text):
        features = []
        features.append(len(resume_text.split()))
        features.append(len(job_text.split()))
        resume_words = set(resume_text.lower().split())
        job_words = set(job_text.lower().split())
        overlap = len(resume_words.intersection(job_words))
        features.append(overlap)
        features.append(overlap / len(job_words) if job_words else 0)
        for category, skills in self.skill_categories.items():
            resume_skills = sum(1 for skill in skills if skill in resume_text.lower())
            job_skills = sum(1 for skill in skills if skill in job_text.lower())
            features.append(resume_skills)
            features.append(job_skills)
            features.append(resume_skills / max(job_skills, 1) if job_skills else 0)
        features.append(1 if 'experience' in resume_text.lower() else 0)
        features.append(1 if 'education' in resume_text.lower() else 0)
        features.append(1 if 'skills' in resume_text.lower() else 0)
        return features
    def _train_models(self, training_data):
        try:
            X = training_data['features']
            y_labels = training_data['labels']
            y_scores = training_data['match_scores']
            y_ats = training_data['ats_scores']
            X_scaled = self.scaler.fit_transform(X)
            self.job_classification_model.fit(X_scaled, y_labels)
            self.score_prediction_model.fit(X_scaled, y_scores)
            ats_threshold = np.median(y_ats)
            ats_labels = (y_ats > ats_threshold).astype(int)
            if len(np.unique(ats_labels)) > 1:
                self.ats_compatibility_model.fit(X_scaled, ats_labels)
            else:
                self.logger.warning("Creating balanced ATS training data")
                balanced_labels = np.concatenate([np.zeros(len(y_ats)//2), np.ones(len(y_ats)//2)])
                if len(balanced_labels) < len(y_ats):
                    balanced_labels = np.concatenate([balanced_labels, [0]])
                self.ats_compatibility_model.fit(X_scaled, balanced_labels[:len(X_scaled)])
            self.logger.info("Models trained successfully")
            self._save_models()
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise
    def _save_models(self):
        try:
            models_to_save = {
                'job_classification_model': self.job_classification_model,
                'score_prediction_model': self.score_prediction_model,
                'ats_compatibility_model': self.ats_compatibility_model,
                'scaler': self.scaler
            }
            for model_name, model in models_to_save.items():
                filename = f"{model_name}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)
            self.logger.info("Models saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    def _load_models(self):
        try:
            model_files = [
                'job_classification_model.pkl',
                'score_prediction_model.pkl',
                'ats_compatibility_model.pkl',
                'scaler.pkl'
            ]
            if all(os.path.exists(f) for f in model_files):
                with open('job_classification_model.pkl', 'rb') as f:
                    self.job_classification_model = pickle.load(f)
                with open('score_prediction_model.pkl', 'rb') as f:
                    self.score_prediction_model = pickle.load(f)
                with open('ats_compatibility_model.pkl', 'rb') as f:
                    self.ats_compatibility_model = pickle.load(f)
                with open('scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                self.models_trained = True
                self.logger.info("Models loaded successfully from disk")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            return False
    def predict_job_category(self, resume_text, job_text):
        if not self.models_trained:
            return "Unknown"
        try:
            features = self._extract_features(resume_text, job_text)
            features_scaled = self.scaler.transform([features])
            prediction = self.job_classification_model.predict(features_scaled)[0]
            confidence = max(self.job_classification_model.predict_proba(features_scaled)[0])
            return {
                'category': prediction,
                'confidence': confidence
            }
        except Exception as e:
            self.logger.error(f"Error predicting job category: {str(e)}")
            return "Unknown"
    def predict_match_score(self, resume_text, job_text):
        if not self.models_trained:
            return 0
        try:
            features = self._extract_features(resume_text, job_text)
            features_scaled = self.scaler.transform([features])
            predicted_score = self.score_prediction_model.predict(features_scaled)[0]
            return max(0, min(100, predicted_score))
        except Exception as e:
            self.logger.error(f"Error predicting match score: {str(e)}")
            return 0
    def predict_ats_compatibility(self, resume_text):
        if not self.models_trained:
            return 0.5
        try:
            features = self._extract_features(resume_text, "")
            features_scaled = self.scaler.transform([features])
            compatibility_prob = self.ats_compatibility_model.predict_proba(features_scaled)[0][1]
            return compatibility_prob
        except Exception as e:
            self.logger.error(f"Error predicting ATS compatibility: {str(e)}")
            return 0.5
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    def extract_keywords(self, text, top_n=20):
        processed_text = self.preprocess_text(text)
        try:
            temp_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = temp_vectorizer.fit_transform([processed_text])
            feature_names = temp_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            return [keyword for keyword, score in keyword_scores[:top_n] if score > 0]
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
            words = processed_text.split()
            word_freq = {}
            for word in words:
                if len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:top_n]]
    def calculate_keyword_match_score(self, resume_text, job_text):
        resume_keywords = set(self.extract_keywords(resume_text))
        job_keywords = set(self.extract_keywords(job_text))
        if not job_keywords:
            return 0, set(), set()
        matching_keywords = resume_keywords.intersection(job_keywords)
        match_score = len(matching_keywords) / len(job_keywords) * 100
        return min(match_score, 100), matching_keywords, job_keywords - matching_keywords
    def calculate_semantic_similarity(self, resume_text, job_text):
        try:
            processed_resume = self.preprocess_text(resume_text)
            processed_job = self.preprocess_text(job_text)
            temp_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = temp_vectorizer.fit_transform([processed_resume, processed_job])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity * 100
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0
    def analyze_skills_match(self, resume_text, job_text):
        resume_lower = resume_text.lower()
        job_lower = job_text.lower()
        skill_analysis = {}
        missing_skills = []
        matched_skills = []
        for category, skills in self.skill_categories.items():
            resume_skills = [skill for skill in skills if skill in resume_lower]
            job_skills = [skill for skill in skills if skill in job_lower]
            if job_skills:
                matched = set(resume_skills).intersection(set(job_skills))
                missing = set(job_skills) - set(resume_skills)
                skill_analysis[category] = {
                    'required': job_skills,
                    'matched': list(matched),
                    'missing': list(missing),
                    'match_percentage': (len(matched) / len(job_skills)) * 100 if job_skills else 0
                }
                matched_skills.extend(matched)
                missing_skills.extend(missing)
        return skill_analysis, matched_skills, missing_skills
    def analyze_career_recommendations(self, resume_text):
        resume_lower = resume_text.lower()
        career_scores = {}
        for field_name, field_info in self.career_fields.items():
            score = 0
            keyword_matches = 0
            for keyword in field_info['keywords']:
                if keyword in resume_lower:
                    keyword_matches += 1
            keyword_score = (keyword_matches / len(field_info['keywords'])) * 50
            skill_score = 0
            skill_details = {}
            for skill_category in field_info['skills']:
                if skill_category in self.skill_categories:
                    category_skills = self.skill_categories[skill_category]
                    matched_skills = [skill for skill in category_skills if skill in resume_lower]
                    if category_skills:
                        category_score = (len(matched_skills) / len(category_skills)) * 100
                        skill_score += category_score
                        skill_details[skill_category] = {
                            'matched': matched_skills,
                            'total': len(category_skills),
                            'score': category_score
                        }
            if field_info['skills']:
                skill_score = skill_score / len(field_info['skills'])
            skill_score = skill_score * 0.5
            total_score = (keyword_score + skill_score) * field_info['weight']
            career_scores[field_name] = {
                'total_score': round(total_score, 1),
                'keyword_score': round(keyword_score, 1),
                'skill_score': round(skill_score, 1),
                'keyword_matches': keyword_matches,
                'total_keywords': len(field_info['keywords']),
                'skill_details': skill_details,
                'weight': field_info['weight']
            }
        sorted_careers = sorted(career_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        top_recommendations = []
        for i, (field_name, scores) in enumerate(sorted_careers[:5]):
            if scores['total_score'] > 0:
                recommendation = {
                    'field': field_name,
                    'match_percentage': min(scores['total_score'], 100),
                    'keyword_matches': scores['keyword_matches'],
                    'total_keywords': scores['total_keywords'],
                    'skill_categories': len(scores['skill_details']),
                    'rank': i + 1
                }
                reasoning = []
                if scores['keyword_score'] > 20:
                    reasoning.append(f"Strong keyword alignment ({scores['keyword_matches']}/{scores['total_keywords']} key terms)")
                if scores['skill_score'] > 20:
                    reasoning.append(f"Relevant technical skills across {len(scores['skill_details'])} categories")
                recommendation['reasoning'] = reasoning if reasoning else ["Basic skill match found"]
                top_recommendations.append(recommendation)
        return top_recommendations
    def generate_suggestions(self, resume_text, job_text, keyword_match_score, 
                           semantic_score, missing_keywords, missing_skills, skill_analysis):
        suggestions = []
        if keyword_match_score < 50:
            missing_kw_list = list(missing_keywords)[:5] if missing_keywords else []
            suggestions.append({
                'type': 'keyword_optimization',
                'title': 'Improve Keyword Matching',
                'description': f'Your resume matches only {keyword_match_score:.1f}% of key terms from the job description.',
                'action': f'Consider incorporating these important keywords: {", ".join(missing_kw_list)}' if missing_kw_list else 'Review the job description and incorporate relevant terms.'
            })
        if semantic_score < 60:
            suggestions.append({
                'type': 'content_alignment',
                'title': 'Enhance Content Relevance',
                'description': f'The overall content alignment is {semantic_score:.1f}%. Your resume content could be more aligned with the job requirements.',
                'action': 'Rewrite sections to better reflect the job responsibilities and requirements using similar language and terminology.'
            })
        high_priority_missing = []
        for category, analysis in skill_analysis.items():
            if analysis['missing'] and analysis['match_percentage'] < 70:
                high_priority_missing.extend(analysis['missing'][:3])
        if high_priority_missing:
            suggestions.append({
                'type': 'skills_enhancement',
                'title': 'Add Missing Technical Skills',
                'description': 'Several important technical skills mentioned in the job description are missing from your resume.',
                'action': f'Consider highlighting or learning these skills: {", ".join(high_priority_missing[:5])}'
            })
        word_count = len(resume_text.split())
        if word_count < 200:
            suggestions.append({
                'type': 'content_length',
                'title': 'Expand Resume Content',
                'description': 'Your resume appears to be quite brief.',
                'action': 'Add more details about your achievements, responsibilities, and quantifiable results.'
            })
        elif word_count > 800:
            suggestions.append({
                'type': 'content_length',
                'title': 'Optimize Resume Length',
                'description': 'Your resume might be too lengthy.',
                'action': 'Focus on the most relevant experiences and achievements for this specific role.'
            })
        if 'experience' not in resume_text.lower() and 'work' not in resume_text.lower():
            suggestions.append({
                'type': 'experience_highlighting',
                'title': 'Highlight Relevant Experience',
                'description': 'Make sure to clearly highlight your work experience.',
                'action': 'Add a dedicated experience section with job titles, companies, dates, and key achievements.'
            })
        return suggestions
    def calculate_ats_score(self, resume_text):
        score = 100
        issues = []
        recommendations = []
        if len(resume_text.strip()) < 100:
            score -= 20
            issues.append("Resume content is too short or may not have been extracted properly")
            recommendations.append("Ensure your resume is in a standard format (PDF or DOCX)")
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+?1?[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        if not re.search(email_pattern, resume_text):
            score -= 10
            issues.append("No email address detected")
            recommendations.append("Include a professional email address")
        if not re.search(phone_pattern, resume_text):
            score -= 5
            issues.append("No phone number detected")
            recommendations.append("Include a phone number with area code")
        ats_headers = [
            'experience', 'work experience', 'employment', 'professional experience',
            'education', 'skills', 'technical skills', 'summary', 'objective',
            'achievements', 'accomplishments', 'certifications', 'projects'
        ]
        found_headers = 0
        text_lower = resume_text.lower()
        for header in ats_headers:
            if header in text_lower:
                found_headers += 1
        if found_headers < 3:
            score -= 15
            issues.append("Missing standard section headers")
            recommendations.append("Use clear section headers like 'Experience', 'Education', 'Skills'")
        words = resume_text.split()
        if len(words) < 200:
            score -= 10
            issues.append("Resume is too brief")
            recommendations.append("Expand your resume with more detailed descriptions")
        elif len(words) > 1000:
            score -= 5
            issues.append("Resume may be too lengthy for ATS parsing")
            recommendations.append("Consider condensing to 1-2 pages")
        special_chars = ['•', '→', '★', '◆', '▪', '※']
        if any(char in resume_text for char in special_chars):
            score -= 8
            issues.append("Contains special characters that may not parse well")
            recommendations.append("Use simple bullet points (-) instead of special characters")
        date_patterns = [
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b',
            r'\b\d{1,2}/\d{4}\b',
            r'\b\d{4}\s*-\s*\d{4}\b',
            r'\b\d{4}\s*–\s*\d{4}\b'
        ]
        date_found = any(re.search(pattern, resume_text, re.IGNORECASE) for pattern in date_patterns)
        if not date_found:
            score -= 5
            issues.append("No clear date formatting detected")
            recommendations.append("Use consistent date format (e.g., 'Jan 2020 - Dec 2022')")
        if 'skill' not in text_lower:
            score -= 10
            issues.append("No dedicated skills section detected")
            recommendations.append("Include a clear 'Skills' section with relevant keywords")
        score = max(0, score)
        return {
            'ats_score': score,
            'ats_issues': issues,
            'ats_recommendations': recommendations,
            'ats_status': 'Excellent' if score >= 90 else 'Good' if score >= 75 else 'Fair' if score >= 60 else 'Needs Improvement'
        }
    def generate_ats_friendly_format(self, resume_text):
        cleaned_text = resume_text
        special_bullets = ['•', '◆', '▪', '★', '→', '※']
        for bullet in special_bullets:
            cleaned_text = cleaned_text.replace(bullet, '-')
        date_replacements = [
            (r'(\w+)\s+(\d{4})', r'\1 \2'),
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\1/\3'),
        ]
        for pattern, replacement in date_replacements:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        formatting_suggestions = {
            'file_format': 'Save as PDF or DOCX format for best ATS compatibility',
            'font_recommendations': 'Use standard fonts: Arial, Calibri, or Times New Roman (10-12pt)',
            'layout_tips': [
                'Use single-column layout',
                'Avoid headers, footers, and text boxes',
                'Use standard section headers (Experience, Education, Skills)',
                'Left-align all text',
                'Use consistent formatting throughout'
            ],
            'content_optimization': [
                'Include relevant keywords from job description',
                'Use standard job titles and company names',
                'Spell out abbreviations the first time',
                'Include quantifiable achievements',
                'Use action verbs to start bullet points'
            ],
            'contact_info': [
                'Place contact information at the top',
                'Include: Full name, phone, email, city/state',
                'Use professional email address',
                'Include LinkedIn profile URL if available'
            ]
        }
        return {
            'cleaned_text': cleaned_text,
            'formatting_suggestions': formatting_suggestions,
            'ats_optimization_tips': [
                'Use keywords from job description naturally throughout resume',
                'Include both acronyms and full terms (e.g., "AI" and "Artificial Intelligence")',
                'Use standard section headers that ATS systems recognize',
                'Avoid graphics, images, and complex formatting',
                'Test your resume by copying and pasting into a plain text editor'
            ]
        }

    def analyze_resume(self, resume_text, job_description):
        try:
            keyword_score, matching_keywords, missing_keywords = self.calculate_keyword_match_score(
                resume_text, job_description
            )
            semantic_score = self.calculate_semantic_similarity(resume_text, job_description)
            skill_analysis, matched_skills, missing_skills = self.analyze_skills_match(
                resume_text, job_description
            )
            ml_predictions = {}
            if self.models_trained:
                try:
                    job_category_pred = self.predict_job_category(resume_text, job_description)
                    ml_predictions['job_category'] = job_category_pred
                    ml_match_score = self.predict_match_score(resume_text, job_description)
                    ml_predictions['ml_match_score'] = ml_match_score
                    ml_ats_compatibility = self.predict_ats_compatibility(resume_text)
                    ml_predictions['ml_ats_compatibility'] = ml_ats_compatibility
                    combined_score = (
                        keyword_score * 0.25 +
                        semantic_score * 0.25 +
                        ml_match_score * 0.30 +
                        (ml_ats_compatibility * 100) * 0.20
                    )
                except Exception as e:
                    self.logger.error(f"Error in ML predictions: {str(e)}")
                    ml_predictions['error'] = str(e)
                    total_skills = len(matched_skills) + len(missing_skills)
                    skills_score = (len(matched_skills) / max(total_skills, 1)) * 100 if total_skills > 0 else 0
                    combined_score = (
                        keyword_score * 0.4 +
                        semantic_score * 0.4 +
                        skills_score * 0.2
                    )
            else:
                total_skills = len(matched_skills) + len(missing_skills)
                skills_score = (len(matched_skills) / max(total_skills, 1)) * 100 if total_skills > 0 else 0
                combined_score = (
                    keyword_score * 0.4 +
                    semantic_score * 0.4 +
                    skills_score * 0.2
                )
            ats_analysis = self.calculate_ats_score(resume_text)
            ats_formatting = self.generate_ats_friendly_format(resume_text)
            career_recommendations = self.analyze_career_recommendations(resume_text)
            suggestions = self.generate_suggestions(
                resume_text, job_description, keyword_score, semantic_score,
                missing_keywords, missing_skills, skill_analysis
            )
            if self.models_trained and 'job_category' in ml_predictions:
                job_category_info = ml_predictions['job_category']
                if isinstance(job_category_info, dict) and job_category_info.get('confidence', 0) > 0.7:
                    suggestions.append({
                        'type': 'ml_recommendation',
                        'title': 'ML-Based Career Fit',
                        'description': f'Our ML model predicts a {job_category_info["confidence"]:.1%} match for {job_category_info["category"]} roles.',
                        'action': f'Consider highlighting skills and experiences relevant to {job_category_info["category"]} positions.'
                    })
            analysis_details = {
                'keyword_matching': {
                    'score': round(keyword_score, 1),
                    'matched_keywords': list(matching_keywords),
                    'missing_keywords': list(missing_keywords)
                },
                'semantic_similarity': {
                    'score': round(semantic_score, 1)
                },
                'skills_analysis': skill_analysis,
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
                'ats_analysis': {
                    'score': ats_analysis['ats_score'],
                    'status': ats_analysis['ats_status'],
                    'issues': ats_analysis['ats_issues'],
                    'recommendations': ats_analysis['ats_recommendations']
                },
                'ats_formatting': ats_formatting,
                'ml_predictions': ml_predictions if self.models_trained else {'note': 'ML models not available'}
            }
            return {
                'overall_score': round(combined_score, 1),
                'ats_score': ats_analysis['ats_score'],
                'ats_status': ats_analysis['ats_status'],
                'ats_issues': ats_analysis['ats_issues'],
                'ats_recommendations': ats_analysis['ats_recommendations'],
                'ats_formatting': ats_formatting,
                'analysis_details': analysis_details,
                'suggestions': suggestions,
                'career_recommendations': career_recommendations,
                'ml_enhanced': self.models_trained,
                'success': True
            }
        except Exception as e:
            self.logger.error(f"Error in resume analysis: {str(e)}")
            return {
                'error': 'Failed to analyze resume. Please check your input and try again.',
                'success': False
            }
