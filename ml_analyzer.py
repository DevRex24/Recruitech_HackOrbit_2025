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
    
    def calculate_keyword_match_score(self, resume_text, job_text):
        """Calculate keyword matching score between resume and job description."""
        resume_keywords = set(self.extract_keywords(resume_text))
        job_keywords = set(self.extract_keywords(job_text))
        
        if not job_keywords:
            return 0, set(), set()
        
        matching_keywords = resume_keywords.intersection(job_keywords)
        match_score = len(matching_keywords) / len(job_keywords) * 100
        
        return min(match_score, 100), matching_keywords, job_keywords - matching_keywords
    
    def calculate_semantic_similarity(self, resume_text, job_text):
        """Calculate semantic similarity using TF-IDF and cosine similarity."""
        try:
            processed_resume = self.preprocess_text(resume_text)
            processed_job = self.preprocess_text(job_text)
            
            tfidf_matrix = self.vectorizer.fit_transform([processed_resume, processed_job])
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity * 100
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0
    
    def analyze_skills_match(self, resume_text, job_text):
        """Analyze technical skills matching between resume and job description."""
        resume_lower = resume_text.lower()
        job_lower = job_text.lower()
        
        skill_analysis = {}
        missing_skills = []
        matched_skills = []
        
        for category, skills in self.skill_categories.items():
            resume_skills = [skill for skill in skills if skill in resume_lower]
            job_skills = [skill for skill in skills if skill in job_lower]
            
            if job_skills: #analyzing the job requirements and skills
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
        """Analyze resume and recommend career fields based on skills and keywords."""
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
        """Generate actionable improvement suggestions."""
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
