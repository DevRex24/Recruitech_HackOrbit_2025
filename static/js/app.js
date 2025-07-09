class ResumeOptimizer {
    constructor() {
        this.form = document.getElementById('analyzeForm');
        this.loadingSection = document.getElementById('loadingSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.errorSection = document.getElementById('errorSection');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.resumeFile = document.getElementById('resumeFile');
        this.resumeText = document.getElementById('resumeText');
        this.toggleTextBtn = document.getElementById('toggleTextInput');
        this.clearFileBtn = document.getElementById('clearFileBtn');
        this.jobTemplateSelect = document.getElementById('jobTemplateSelect');
        this.loadTemplateBtn = document.getElementById('loadTemplateBtn');
        this.jobDescription = document.getElementById('jobDescription');
        
        this.initializeEventListeners();
        this.loadJobTemplates();
    }
    
    initializeEventListeners() {
        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeResume();
        });
        
        // File upload handling
        this.resumeFile.addEventListener('change', (e) => {
            this.handleFileSelection(e);
        });
        
        // Toggle text input
        this.toggleTextBtn.addEventListener('click', () => {
            this.toggleTextInput();
        });
        
        // Clear file selection
        this.clearFileBtn.addEventListener('click', () => {
            this.clearFileSelection();
        });
        
        // Job template selection
        this.jobTemplateSelect.addEventListener('change', () => {
            this.handleTemplateSelection();
        });
        
        // Load template button
        this.loadTemplateBtn.addEventListener('click', () => {
            this.loadSelectedTemplate();
        });
        
        // Auto-resize textareas
        const textareas = document.querySelectorAll('textarea');
        textareas.forEach(textarea => {
            textarea.addEventListener('input', this.autoResize);
        });
    }
    
    async loadJobTemplates() {
        try {
            const response = await fetch('/job-templates');
            const data = await response.json();
            
            // Populate the dropdown
            Object.entries(data.templates).forEach(([key, template]) => {
                const option = document.createElement('option');
                option.value = key;
                option.textContent = template.title;
                this.jobTemplateSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading job templates:', error);
        }
    }
    
    handleTemplateSelection() {
        const selectedValue = this.jobTemplateSelect.value;
        if (selectedValue) {
            this.loadTemplateBtn.disabled = false;
        } else {
            this.loadTemplateBtn.disabled = true;
        }
    }
    
    async loadSelectedTemplate() {
        const selectedTemplate = this.jobTemplateSelect.value;
        if (!selectedTemplate) return;
        
        try {
            const response = await fetch(`/job-templates/${selectedTemplate}`);
            const template = await response.json();
            
            if (response.ok) {
                this.jobDescription.value = template.description;
                this.jobDescription.style.height = 'auto';
                this.jobDescription.style.height = this.jobDescription.scrollHeight + 'px';
                
                // Add visual feedback for successful template load
                this.jobDescription.classList.add('template-loaded');
                setTimeout(() => {
                    this.jobDescription.classList.remove('template-loaded');
                }, 2000);
                
                // Clear any previous results when loading a new template
                this.hideAllSections();
            } else {
                this.showError('Failed to load job template. Please try again.');
            }
        } catch (error) {
            console.error('Error loading template:', error);
            this.showError('Failed to load job template. Please try again.');
        }
    }
    
    handleFileSelection(e) {
        const file = e.target.files[0];
        if (file) {
            // Hide text input when file is selected
            this.resumeText.style.display = 'none';
            this.toggleTextBtn.innerHTML = '<i class="fas fa-edit me-1"></i>Use Text Input';
            this.clearFileBtn.style.display = 'block';
            
            // Validate file size (16MB limit)
            if (file.size > 16 * 1024 * 1024) {
                this.showError('File size exceeds 16MB limit. Please choose a smaller file.');
                this.clearFileSelection();
                return;
            }
            
            // Validate file type
            const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
            if (!allowedTypes.includes(file.type) && !file.name.match(/\.(pdf|docx|txt)$/i)) {
                this.showError('Invalid file type. Please upload a PDF, DOCX, or TXT file.');
                this.clearFileSelection();
                return;
            }
        }
    }
    
    clearFileSelection() {
        this.resumeFile.value = '';
        this.clearFileBtn.style.display = 'none';
        this.hideAllSections();
    }
    
    toggleTextInput() {
        if (this.resumeText.style.display === 'none') {
            // Show text input
            this.resumeText.style.display = 'block';
            this.toggleTextBtn.innerHTML = '<i class="fas fa-eye-slash me-1"></i>Hide Text Input';
            // Clear file selection when switching to text
            this.clearFileSelection();
        } else {
            // Hide text input
            this.resumeText.style.display = 'none';
            this.toggleTextBtn.innerHTML = '<i class="fas fa-edit me-1"></i>Use Text Input';
        }
    }
    
    autoResize(e) {
        e.target.style.height = 'auto';
        e.target.style.height = e.target.scrollHeight + 'px';
    }
    
    async analyzeResume() {
        const jobDescription = document.getElementById('jobDescription').value.trim();
        const resumeFile = this.resumeFile.files[0];
        const resumeText = this.resumeText.value.trim();
        
        // Validate inputs
        if (!jobDescription) {
            this.showError('Please provide a job description.');
            return;
        }
        
        if (!resumeFile && !resumeText) {
            this.showError('Please either upload a resume file or enter resume text.');
            return;
        }
        
        this.showLoading();
        
        try {
            let response;
            
            if (resumeFile) {
                // Handle file upload
                const formData = new FormData();
                formData.append('resume_file', resumeFile);
                formData.append('job_description', jobDescription);
                
                response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
            } else {
                // Handle text input
                response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        resume: resumeText,
                        job_description: jobDescription
                    })
                });
            }
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Analysis failed');
            }
            
            if (data.success) {
                this.showResults(data);
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(error.message || 'An error occurred while analyzing your resume. Please try again.');
        }
    }
    
    showLoading() {
        this.hideAllSections();
        this.loadingSection.classList.remove('d-none');
        this.analyzeBtn.disabled = true;
        this.analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
    }
    
    showResults(data) {
        this.hideAllSections();
        this.resultsSection.classList.remove('d-none');
        
        // Update overall score
        this.updateScoreDisplay(data.overall_score);
        
        // Update ATS score in detailed analysis section
        if (data.ats_score !== undefined) {
            document.getElementById('atsScore').textContent = `${data.ats_score}%`;
            const atsStatusElement = document.getElementById('atsStatus');
            atsStatusElement.textContent = data.ats_status;
            
            // Set ATS status badge color
            atsStatusElement.className = 'badge';
            if (data.ats_score >= 90) {
                atsStatusElement.classList.add('bg-success');
            } else if (data.ats_score >= 75) {
                atsStatusElement.classList.add('bg-info');
            } else if (data.ats_score >= 60) {
                atsStatusElement.classList.add('bg-warning');
            } else {
                atsStatusElement.classList.add('bg-danger');
            }
        }
        
        // Update detailed analysis
        this.updateDetailedAnalysis(data.analysis_details);
        
        // Update suggestions
        this.updateSuggestions(data.suggestions);
        
        // Update skills analysis
        this.updateSkillsAnalysis(data.analysis_details.skills_analysis);
        
        // Update ATS analysis
        this.updateATSAnalysis(data);
        
        // Update ATS formatting suggestions
        this.updateATSFormatting(data.ats_formatting);
        
        // Update career recommendations
        this.updateCareerRecommendations(data.career_recommendations);
        
        this.resetButton();
    }
    
    showError(message) {
        this.hideAllSections();
        this.errorSection.classList.remove('d-none');
        document.getElementById('errorMessage').textContent = message;
        this.resetButton();
    }
    
    hideAllSections() {
        this.loadingSection.classList.add('d-none');
        this.resultsSection.classList.add('d-none');
        this.errorSection.classList.add('d-none');
    }
    
    resetButton() {
        this.analyzeBtn.disabled = false;
        this.analyzeBtn.innerHTML = '<i class="fas fa-chart-line me-2"></i>Analyze Resume Fit';
    }
    
    updateScoreDisplay(score) {
        const scoreText = document.getElementById('scoreText');
        const scoreCircle = document.getElementById('scoreCircle');
        const scoreTitle = document.getElementById('scoreTitle');
        
        scoreText.textContent = `${score}%`;
        
        // Update circle color based on score
        scoreCircle.className = 'progress-circle';
        if (score >= 80) {
            scoreCircle.classList.add('score-excellent');
            scoreTitle.textContent = 'Excellent Match!';
        } else if (score >= 60) {
            scoreCircle.classList.add('score-good');
            scoreTitle.textContent = 'Good Match';
        } else if (score >= 40) {
            scoreCircle.classList.add('score-fair');
            scoreTitle.textContent = 'Fair Match';
        } else {
            scoreCircle.classList.add('score-poor');
            scoreTitle.textContent = 'Needs Improvement';
        }
        
        // Animate the score
        this.animateScore(scoreText, score);
    }
    
    animateScore(element, targetScore) {
        let currentScore = 0;
        const increment = targetScore / 50;
        const timer = setInterval(() => {
            currentScore += increment;
            if (currentScore >= targetScore) {
                currentScore = targetScore;
                clearInterval(timer);
            }
            element.textContent = `${Math.round(currentScore)}%`;
        }, 20);
    }
    
    updateDetailedAnalysis(analysisDetails) {
        document.getElementById('keywordScore').textContent = `${analysisDetails.keyword_matching.score}%`;
        document.getElementById('semanticScore').textContent = `${analysisDetails.semantic_similarity.score}%`;
        
        const matchedSkills = analysisDetails.matched_skills || [];
        const missingSkills = analysisDetails.missing_skills || [];
        const totalSkills = matchedSkills.length + missingSkills.length;
        
        document.getElementById('skillsMatch').textContent = `${matchedSkills.length}/${totalSkills}`;
    }
    
    updateSuggestions(suggestions) {
        const suggestionsList = document.getElementById('suggestionsList');
        
        if (!suggestions || suggestions.length === 0) {
            suggestionsList.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle me-2"></i>
                    Great job! Your resume is well-optimized for this job description.
                </div>
            `;
            return;
        }
        
        const suggestionsHTML = suggestions.map((suggestion, index) => {
            const iconMap = {
                'keyword_optimization': 'fas fa-key',
                'content_alignment': 'fas fa-brain',
                'skills_enhancement': 'fas fa-cogs',
                'content_length': 'fas fa-file-alt',
                'experience_highlighting': 'fas fa-briefcase'
            };
            
            const icon = iconMap[suggestion.type] || 'fas fa-lightbulb';
            
            return `
                <div class="card mb-3">
                    <div class="card-body">
                        <div class="d-flex align-items-start">
                            <div class="flex-shrink-0">
                                <div class="bg-warning bg-opacity-10 rounded-circle d-flex align-items-center justify-content-center" style="width: 40px; height: 40px;">
                                    <i class="${icon} text-warning"></i>
                                </div>
                            </div>
                            <div class="flex-grow-1 ms-3">
                                <h6 class="card-title mb-2">${suggestion.title}</h6>
                                <p class="card-text text-muted mb-2">${suggestion.description}</p>
                                <div class="alert alert-info mb-0">
                                    <strong>Action:</strong> ${suggestion.action}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        suggestionsList.innerHTML = suggestionsHTML;
    }
    
    updateSkillsAnalysis(skillsAnalysis) {
        const skillsAnalysisDiv = document.getElementById('skillsAnalysis');
        
        if (!skillsAnalysis || Object.keys(skillsAnalysis).length === 0) {
            skillsAnalysisDiv.innerHTML = `
                <p class="text-muted">No specific technical skills analysis available.</p>
            `;
            return;
        }
        
        const skillsHTML = Object.entries(skillsAnalysis).map(([category, analysis]) => {
            if (!analysis.required || analysis.required.length === 0) {
                return '';
            }
            
            const matchPercentage = analysis.match_percentage || 0;
            const progressBarClass = matchPercentage >= 70 ? 'bg-success' : 
                                   matchPercentage >= 40 ? 'bg-warning' : 'bg-danger';
            
            return `
                <div class="mb-4">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h6 class="mb-0 text-capitalize">${category.replace('_', ' ')}</h6>
                        <span class="badge bg-secondary">${Math.round(matchPercentage)}% Match</span>
                    </div>
                    
                    <div class="progress mb-2" style="height: 6px;">
                        <div class="progress-bar ${progressBarClass}" role="progressbar" 
                             style="width: ${matchPercentage}%" 
                             aria-valuenow="${matchPercentage}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                    
                    <div class="row">
                        ${analysis.matched.length > 0 ? `
                        <div class="col-md-6">
                            <small class="text-success fw-bold">✓ Matched Skills:</small>
                            <div class="mt-1">
                                ${analysis.matched.map(skill => 
                                    `<span class="badge bg-success bg-opacity-20 text-success me-1 mb-1">${skill}</span>`
                                ).join('')}
                            </div>
                        </div>
                        ` : ''}
                        
                        ${analysis.missing.length > 0 ? `
                        <div class="col-md-6">
                            <small class="text-danger fw-bold">✗ Missing Skills:</small>
                            <div class="mt-1">
                                ${analysis.missing.map(skill => 
                                    `<span class="badge bg-danger bg-opacity-20 text-danger me-1 mb-1">${skill}</span>`
                                ).join('')}
                            </div>
                        </div>
                        ` : ''}
                    </div>
                </div>
            `;
        }).filter(html => html !== '').join('');
        
        if (skillsHTML) {
            skillsAnalysisDiv.innerHTML = skillsHTML;
        } else {
            skillsAnalysisDiv.innerHTML = `
                <p class="text-muted">No technical skills were identified in the job description for analysis.</p>
            `;
        }
    }
    
    updateCareerRecommendations(recommendations) {
        const careerDiv = document.getElementById('careerRecommendations');
        
        if (!recommendations || recommendations.length === 0) {
            careerDiv.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    No specific career field recommendations could be generated from your resume content.
                </div>
            `;
            return;
        }
        
        const careerHTML = `
            <div class="mb-3">
                <p class="text-muted">Based on your resume content, here are the career fields that best match your skills and experience:</p>
            </div>
            ${recommendations.map((rec, index) => {
                const badgeClass = index === 0 ? 'bg-success' : index === 1 ? 'bg-info' : 'bg-secondary';
                const rankIcon = index === 0 ? 'fas fa-crown' : index === 1 ? 'fas fa-medal' : 'fas fa-star';
                
                return `
                    <div class="card mb-3 border-start border-4 ${index === 0 ? 'border-success' : index === 1 ? 'border-info' : 'border-secondary'}">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <div class="d-flex align-items-center">
                                    <i class="${rankIcon} text-${index === 0 ? 'success' : index === 1 ? 'info' : 'secondary'} me-2"></i>
                                    <h6 class="card-title mb-0">${rec.field}</h6>
                                </div>
                                <div class="d-flex align-items-center">
                                    <span class="badge ${badgeClass} me-2">#${rec.rank}</span>
                                    <span class="badge bg-primary">${Math.round(rec.match_percentage)}% Match</span>
                                </div>
                            </div>
                            
                            <div class="progress mb-3" style="height: 8px;">
                                <div class="progress-bar ${index === 0 ? 'bg-success' : index === 1 ? 'bg-info' : 'bg-secondary'}" 
                                     role="progressbar" 
                                     style="width: ${rec.match_percentage}%" 
                                     aria-valuenow="${rec.match_percentage}" 
                                     aria-valuemin="0" 
                                     aria-valuemax="100">
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-8">
                                    <strong>Why this field matches:</strong>
                                    <ul class="list-unstyled mt-2">
                                        ${rec.reasoning.map(reason => `
                                            <li class="mb-1">
                                                <i class="fas fa-check-circle text-success me-2"></i>
                                                ${reason}
                                            </li>
                                        `).join('')}
                                    </ul>
                                </div>
                                <div class="col-md-4">
                                    <div class="text-end">
                                        <small class="text-muted">
                                            Keywords: ${rec.keyword_matches}/${rec.total_keywords}<br>
                                            Skill Areas: ${rec.skill_categories}
                                        </small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('')}
            
            <div class="alert alert-light mt-3">
                <i class="fas fa-lightbulb me-2"></i>
                <strong>Career Guidance:</strong> These recommendations are based on the skills and keywords in your resume. 
                Consider exploring job opportunities in your top-ranked fields or developing skills for fields you're interested in pursuing.
            </div>
        `;
        
        careerDiv.innerHTML = careerHTML;
    }
    
    updateATSAnalysis(data) {
        const atsDiv = document.getElementById('atsAnalysis');
        
        if (!data.ats_score || data.ats_score === undefined) {
            atsDiv.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    ATS analysis could not be completed.
                </div>
            `;
            return;
        }
        
        const atsHTML = `
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card border-0 bg-light">
                        <div class="card-body text-center">
                            <div class="display-4 mb-2 ${data.ats_score >= 75 ? 'text-success' : data.ats_score >= 60 ? 'text-warning' : 'text-danger'}">
                                ${data.ats_score}%
                            </div>
                            <h6 class="text-muted">ATS Compatibility Score</h6>
                            <span class="badge ${data.ats_score >= 90 ? 'bg-success' : data.ats_score >= 75 ? 'bg-info' : data.ats_score >= 60 ? 'bg-warning' : 'bg-danger'}">
                                ${data.ats_status}
                            </span>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card border-0 bg-light">
                        <div class="card-body">
                            <h6 class="card-title">What this means:</h6>
                            <p class="card-text small mb-0">
                                ${data.ats_score >= 90 ? 'Your resume is highly compatible with ATS systems and should parse well.' :
                                  data.ats_score >= 75 ? 'Your resume has good ATS compatibility with minor improvements needed.' :
                                  data.ats_score >= 60 ? 'Your resume may have some ATS compatibility issues that should be addressed.' :
                                  'Your resume has significant ATS compatibility issues that need attention.'}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            
            ${data.ats_issues && data.ats_issues.length > 0 ? `
                <div class="mb-4">
                    <h6 class="text-warning"><i class="fas fa-exclamation-triangle me-2"></i>Issues Found:</h6>
                    <ul class="list-group list-group-flush">
                        ${data.ats_issues.map(issue => `
                            <li class="list-group-item border-0 bg-transparent">
                                <i class="fas fa-times text-danger me-2"></i>
                                ${issue}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${data.ats_recommendations && data.ats_recommendations.length > 0 ? `
                <div class="mb-4">
                    <h6 class="text-info"><i class="fas fa-lightbulb me-2"></i>Recommendations:</h6>
                    <ul class="list-group list-group-flush">
                        ${data.ats_recommendations.map(rec => `
                            <li class="list-group-item border-0 bg-transparent">
                                <i class="fas fa-check text-success me-2"></i>
                                ${rec}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}
        `;
        
        atsDiv.innerHTML = atsHTML;
    }
    
    updateATSFormatting(atsFormatting) {
        const formattingDiv = document.getElementById('atsFormatting');
        
        if (!atsFormatting || !atsFormatting.formatting_suggestions) {
            formattingDiv.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    ATS formatting suggestions could not be generated.
                </div>
            `;
            return;
        }
        
        const suggestions = atsFormatting.formatting_suggestions;
        const optimizationTips = atsFormatting.ats_optimization_tips || [];
        
        const formattingHTML = `
            <div class="row">
                <div class="col-md-6 mb-4">
                    <div class="card h-100">
                        <div class="card-header bg-primary text-white">
                            <h6 class="mb-0"><i class="fas fa-file-alt me-2"></i>File Format & Fonts</h6>
                        </div>
                        <div class="card-body">
                            <p class="small mb-2"><strong>File Format:</strong></p>
                            <p class="small text-muted mb-3">${suggestions.file_format}</p>
                            <p class="small mb-2"><strong>Font Recommendations:</strong></p>
                            <p class="small text-muted">${suggestions.font_recommendations}</p>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6 mb-4">
                    <div class="card h-100">
                        <div class="card-header bg-info text-white">
                            <h6 class="mb-0"><i class="fas fa-layout me-2"></i>Layout Guidelines</h6>
                        </div>
                        <div class="card-body">
                            <ul class="list-unstyled small mb-0">
                                ${suggestions.layout_tips.map(tip => `
                                    <li class="mb-1">
                                        <i class="fas fa-check text-success me-2"></i>
                                        ${tip}
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6 mb-4">
                    <div class="card h-100">
                        <div class="card-header bg-success text-white">
                            <h6 class="mb-0"><i class="fas fa-edit me-2"></i>Content Optimization</h6>
                        </div>
                        <div class="card-body">
                            <ul class="list-unstyled small mb-0">
                                ${suggestions.content_optimization.map(tip => `
                                    <li class="mb-1">
                                        <i class="fas fa-arrow-right text-primary me-2"></i>
                                        ${tip}
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6 mb-4">
                    <div class="card h-100">
                        <div class="card-header bg-warning text-dark">
                            <h6 class="mb-0"><i class="fas fa-address-card me-2"></i>Contact Information</h6>
                        </div>
                        <div class="card-body">
                            <ul class="list-unstyled small mb-0">
                                ${suggestions.contact_info.map(tip => `
                                    <li class="mb-1">
                                        <i class="fas fa-user text-info me-2"></i>
                                        ${tip}
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            ${optimizationTips.length > 0 ? `
                <div class="alert alert-light border">
                    <h6 class="alert-heading">
                        <i class="fas fa-robot me-2"></i>
                        ATS Optimization Tips
                    </h6>
                    <ul class="mb-0 small">
                        ${optimizationTips.map(tip => `
                            <li class="mb-1">${tip}</li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${atsFormatting.cleaned_text ? `
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0">
                            <i class="fas fa-download me-2"></i>
                            ATS-Optimized Resume Text
                            <button class="btn btn-sm btn-outline-primary float-end" onclick="navigator.clipboard.writeText(document.getElementById('cleanedText').textContent)">
                                <i class="fas fa-copy me-1"></i>Copy Text
                            </button>
                        </h6>
                    </div>
                    <div class="card-body">
                        <p class="small text-muted mb-2">Below is your resume text optimized for ATS compatibility:</p>
                        <pre id="cleanedText" class="bg-light p-3 small" style="max-height: 300px; overflow-y: auto;">${atsFormatting.cleaned_text}</pre>
                    </div>
                </div>
            ` : ''}
        `;
        
        formattingDiv.innerHTML = formattingHTML;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ResumeOptimizer();
});
