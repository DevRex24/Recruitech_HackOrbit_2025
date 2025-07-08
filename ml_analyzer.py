import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import string

class ResumeAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.logger = logging.getLogger(__name__)
        
        # Technical skills categories for better analysis
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
        
        # Career field mappings based on skills and keywords
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
        
    def preprocess_text(self, text):
        """Clean and preprocess text for analysis."""
        text = text.lower()
        
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        text = ' '.join(text.split())
        
        return text
    
    def extract_keywords(self, text, top_n=20):
        """Extract important keywords from text using TF-IDF."""
        processed_text = self.preprocess_text(text)
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform([processed_text])
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Keywords Extraction
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