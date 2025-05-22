/**
 * Citation Recommendation System - Frontend JavaScript
 * 
 * This script handles the UI interactions for the citation recommendation system,
 * including form submissions, dynamic elements, and API calls.
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const loadExampleBtn = document.getElementById('load-example');
    const addAuthorBtn = document.getElementById('add-author');
    const authorsContainer = document.getElementById('authors-container');
    const addReferenceBtn = document.getElementById('add-reference');
    const referencesContainer = document.getElementById('references-container');
    const submitBtn = document.getElementById('submit-btn');
    const resultsContainer = document.getElementById('results-container');
    const spinner = document.getElementById('spinner');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    const observedRatioInput = document.getElementById('observed_ratio');
    const observedRatioValue = document.getElementById('observed_ratio_value');
    
    // Initialize
    init();
    
    /**
     * Initialize the application
     */
    function init() {
        // Setup event listeners
        setupEventListeners();
        
        // Update observed ratio display
        updateObservedRatio();
    }
    
    /**
     * Setup all event listeners
     */
    function setupEventListeners() {
        // Load example paper
        loadExampleBtn.addEventListener('click', loadExamplePaper);
        
        // Add/remove authors
        addAuthorBtn.addEventListener('click', addAuthor);
        document.addEventListener('click', function(e) {
            if (e.target && e.target.classList.contains('remove-author')) {
                removeAuthor(e.target);
            }
        });
        
        // Add/remove references
        addReferenceBtn.addEventListener('click', addReference);
        document.addEventListener('click', function(e) {
            if (e.target && e.target.classList.contains('remove-reference')) {
                removeReference(e.target.closest('.reference-item'));
            }
            if (e.target && e.target.classList.contains('add-reference-author')) {
                addReferenceAuthor(e.target.closest('.reference-item'));
            }
            if (e.target && e.target.classList.contains('remove-reference-author')) {
                removeReferenceAuthor(e.target);
            }
        });
        
        // Submit form
        submitBtn.addEventListener('click', submitForm);
        
        // Observed ratio slider
        observedRatioInput.addEventListener('input', updateObservedRatio);
    }
    
    /**
     * Load example paper from server
     */
    function loadExamplePaper() {
        // Show spinner
        spinner.classList.remove('d-none');
        
        // Clear any previous results and errors
        resultsContainer.classList.add('d-none');
        errorContainer.classList.add('d-none');
        
        // Fetch example paper from server
        fetch('/load_example')
            .then(response => response.json())
            .then(paper => {
                let redirectTab = null;
                let alertMessage = null;
                
                // Track fields that need attention due to null values
                const nullFields = [];
                
                // Populate form with paper data
                document.getElementById('title').value = paper.title || '';
                if (!paper.title) nullFields.push('title');
                
                // Handle publication date
                if (paper.published_date) {
                    document.getElementById('published_date').value = formatDate(paper.published_date);
                } else {
                    document.getElementById('published_date').value = '';
                    nullFields.push('published date');
                }
                
                // Handle potentially null fields
                document.getElementById('journal').value = paper.journal || '';
                if (!paper.journal) nullFields.push('journal');
                
                document.getElementById('doi').value = paper.doi || '';
                if (!paper.doi) nullFields.push('DOI');
                
                document.getElementById('abstract').value = paper.abstract ? 
                    paper.abstract.replace(/<[^>]*>/g, '') : ''; // Remove HTML tags
                if (!paper.abstract) nullFields.push('abstract');
                
                // Check for volume and issue (if they exist as properties in the paper object)
                if (paper.hasOwnProperty('volume') && paper.volume === null) nullFields.push('volume');
                if (paper.hasOwnProperty('issue') && paper.issue === null) nullFields.push('issue');
                
                // Clear existing authors
                authorsContainer.innerHTML = '';
                
                // Add authors
                if (paper.authors && paper.authors.length > 0) {
                    let hasNullAuthorData = false;
                    
                    paper.authors.forEach(author => {
                        const authorRow = createAuthorRow();
                        const inputs = authorRow.querySelectorAll('input');
                        
                        // Check for null values in author fields
                        if (author[0] === null || author[1] === null) {
                            hasNullAuthorData = true;
                        }
                        
                        inputs[0].value = author[0] || ''; // First name
                        inputs[1].value = author[1] || ''; // Last name
                        authorsContainer.appendChild(authorRow);
                    });
                    
                    if (hasNullAuthorData) {
                        nullFields.push('author information');
                    }
                } else {
                    // Add at least one empty author row
                    authorsContainer.appendChild(createAuthorRow());
                    nullFields.push('authors');
                }
                
                // Clear existing references
                referencesContainer.innerHTML = '';
                
                // Add references
                if (paper.references && paper.references.length > 0) {
                    let hasNullReferenceData = false;
                    
                    paper.references.forEach(ref => {
                        const refElement = addReference();
                        
                        // Set reference type
                        const typeSelect = refElement.querySelector('.reference-type');
                        if (ref.reference_type) {
                            typeSelect.value = ref.reference_type;
                        }
                        
                        // Check for null values in reference fields
                        if (ref.title === null || ref.year === null || ref.journal === null || ref.doi === null) {
                            hasNullReferenceData = true;
                        }
                        
                        // Set reference details
                        refElement.querySelector('.reference-title').value = ref.title || '';
                        refElement.querySelector('.reference-year').value = ref.year || '';
                        refElement.querySelector('.reference-journal').value = ref.journal || '';
                        refElement.querySelector('.reference-doi').value = ref.doi || '';
                        
                        // Clear existing reference authors
                        const refAuthorsContainer = refElement.querySelector('.reference-authors-container');
                        refAuthorsContainer.innerHTML = '';
                        
                        // Add reference authors
                        if (ref.authors && ref.authors.length > 0) {
                            let hasNullAuthorData = false;
                            
                            ref.authors.forEach(author => {
                                const authorRow = createReferenceAuthorRow();
                                const inputs = authorRow.querySelectorAll('input');
                                
                                // Check for null values in author fields
                                if (author[0] === null || author[1] === null) {
                                    hasNullAuthorData = true;
                                }
                                
                                inputs[0].value = author[0] || ''; // First name
                                inputs[1].value = author[1] || ''; // Last name
                                refAuthorsContainer.appendChild(authorRow);
                            });
                            
                            if (hasNullAuthorData) {
                                hasNullReferenceData = true;
                            }
                        } else {
                            // Add at least one empty author row
                            refAuthorsContainer.appendChild(createReferenceAuthorRow());
                            hasNullReferenceData = true;
                        }
                    });
                    
                    if (hasNullReferenceData) {
                        nullFields.push('reference information');
                    }
                }
                
                // Determine which tab to activate based on null fields
                if (nullFields.length > 0) {
                    if (nullFields.includes('title') || 
                        nullFields.includes('published date') || 
                        nullFields.includes('journal') || 
                        nullFields.includes('DOI') || 
                        nullFields.includes('abstract') || 
                        nullFields.includes('volume') || 
                        nullFields.includes('issue') || 
                        nullFields.includes('author information') ||
                        nullFields.includes('authors')) {
                        
                        // Activate the paper details tab
                        redirectTab = 'paper-tab';
                        
                    } else if (nullFields.includes('reference information')) {
                        
                        // Activate the references tab
                        redirectTab = 'references-tab';
                    }
                    
                    // Create alert message
                    alertMessage = 'Some fields in the example paper have null values: ' + 
                                   nullFields.join(', ') + 
                                   '. Please review and complete these fields.';
                }
                
                // Hide spinner
                spinner.classList.add('d-none');
                
                // Switch to the appropriate tab if needed
                if (redirectTab) {
                    document.getElementById(redirectTab).click();
                    
                    // Show alert message
                    errorMessage.textContent = alertMessage;
                    errorContainer.classList.remove('d-none');
                    errorContainer.classList.remove('alert-danger');
                    errorContainer.classList.add('alert-warning');
                    
                    // Scroll to top of form
                    document.querySelector('.card-header').scrollIntoView({ behavior: 'smooth' });
                }
            })
            .catch(error => {
                console.error('Error loading example paper:', error);
                errorMessage.textContent = 'Failed to load example paper. Please try again.';
                errorContainer.classList.remove('d-none');
                errorContainer.classList.add('alert-danger');
                errorContainer.classList.remove('alert-warning');
                spinner.classList.add('d-none');
            });
    }
    
    /**
     * Add a new author row
     */
    function addAuthor() {
        const authorRow = createAuthorRow();
        authorsContainer.appendChild(authorRow);
    }
    
    /**
     * Create an author row element
     */
    function createAuthorRow() {
        const row = document.createElement('div');
        row.className = 'row author-row mb-2';
        row.innerHTML = `
            <div class="col-md-5">
                <input type="text" class="form-control" name="first_name" placeholder="First Name" required>
            </div>
            <div class="col-md-5">
                <input type="text" class="form-control" name="last_name" placeholder="Last Name" required>
            </div>
            <div class="col-md-2">
                <button type="button" class="btn btn-danger remove-author">Remove</button>
            </div>
        `;
        return row;
    }
    
    /**
     * Remove an author row
     */
    function removeAuthor(button) {
        const row = button.closest('.author-row');
        
        // Don't remove if it's the last author
        if (document.querySelectorAll('.author-row').length > 1) {
            row.remove();
        }
    }
    
    /**
     * Add a new reference item
     */
    function addReference() {
        const template = document.getElementById('reference-template');
        const clone = document.importNode(template.content, true);
        const referenceItem = clone.querySelector('.reference-item');
        referencesContainer.appendChild(referenceItem);
        return referenceItem;
    }
    
    /**
     * Remove a reference item
     */
    function removeReference(referenceItem) {
        referenceItem.remove();
    }
    
    /**
     * Add a new author row to a reference
     */
    function addReferenceAuthor(referenceItem) {
        const authorRow = createReferenceAuthorRow();
        const container = referenceItem.querySelector('.reference-authors-container');
        container.appendChild(authorRow);
    }
    
    /**
     * Create a reference author row element
     */
    function createReferenceAuthorRow() {
        const row = document.createElement('div');
        row.className = 'row reference-author-row mb-2';
        row.innerHTML = `
            <div class="col-md-5">
                <input type="text" class="form-control" placeholder="First Name">
            </div>
            <div class="col-md-5">
                <input type="text" class="form-control" placeholder="Last Name">
            </div>
            <div class="col-md-2">
                <button type="button" class="btn btn-danger remove-reference-author">Remove</button>
            </div>
        `;
        return row;
    }
    
    /**
     * Remove a reference author row
     */
    function removeReferenceAuthor(button) {
        const row = button.closest('.reference-author-row');
        const container = row.closest('.reference-authors-container');
        
        // Don't remove if it's the last author
        if (container.querySelectorAll('.reference-author-row').length > 1) {
            row.remove();
        }
    }
    
    /**
     * Update observed ratio value display
     */
    function updateObservedRatio() {
        observedRatioValue.textContent = observedRatioInput.value;
    }
    
    /**
     * Submit the form data to the server
     */
    function submitForm() {
        // Validate form
        if (!validateForm()) {
            return;
        }
        
        // Clear any previous results and errors
        resultsContainer.classList.add('d-none');
        errorContainer.classList.add('d-none');
        
        // Show spinner
        spinner.classList.remove('d-none');
        
        // Collect form data
        const formData = collectFormData();
        
        // Make API request
        fetch('/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
            .then(response => response.json())
            .then(data => {
                // Handle response
                if (data.error) {
                    // Show error message
                    errorMessage.textContent = data.error;
                    errorContainer.classList.remove('d-none');
                    errorContainer.classList.add('alert-danger');
                    errorContainer.classList.remove('alert-warning');
                } else {
                    // Display results
                    displayResults(data);
                    resultsContainer.classList.remove('d-none');
                }
                
                // Hide spinner
                spinner.classList.add('d-none');
            })
            .catch(error => {
                console.error('Error submitting form:', error);
                errorMessage.textContent = 'An error occurred while processing your request. Please try again.';
                errorContainer.classList.remove('d-none');
                errorContainer.classList.add('alert-danger');
                errorContainer.classList.remove('alert-warning');
                spinner.classList.add('d-none');
            });
    }
    
    /**
     * Validate form data
     */
    function validateForm() {
        // Reset error container styling
        errorContainer.classList.add('alert-danger');
        errorContainer.classList.remove('alert-warning');
        
        // Validate title
        const title = document.getElementById('title').value.trim();
        if (!title) {
            showError('Paper title is required');
            document.getElementById('paper-tab').click(); // Switch to paper tab
            document.getElementById('title').focus();
            return false;
        }
        
        // Validate publication date
        const publishedDate = document.getElementById('published_date').value.trim();
        if (!publishedDate) {
            showError('Publication date is required');
            document.getElementById('paper-tab').click(); // Switch to paper tab
            document.getElementById('published_date').focus();
            return false;
        }
        
        // Validate authors (at least one author with first and last name)
        const authorRows = document.querySelectorAll('.author-row');
        let hasValidAuthor = false;
        
        for (const row of authorRows) {
            const inputs = row.querySelectorAll('input');
            if (inputs[0].value.trim() && inputs[1].value.trim()) {
                hasValidAuthor = true;
                break;
            }
        }
        
        if (!hasValidAuthor) {
            showError('At least one author with first and last name is required');
            document.getElementById('paper-tab').click(); // Switch to paper tab
            document.querySelector('.author-row input').focus();
            return false;
        }
        
        // Validate references (at least title is required)
        const referenceItems = document.querySelectorAll('.reference-item');
        for (const item of referenceItems) {
            const titleInput = item.querySelector('.reference-title');
            if (!titleInput.value.trim()) {
                showError('All references must have a title');
                document.getElementById('references-tab').click(); // Switch to references tab
                titleInput.focus();
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Show an error message
     */
    function showError(message) {
        errorMessage.textContent = message;
        errorContainer.classList.remove('d-none');
        
        // Update error heading based on alert type
        const errorHeading = document.getElementById('error-heading');
        if (errorContainer.classList.contains('alert-warning')) {
            errorHeading.textContent = 'Attention:';
        } else {
            errorHeading.textContent = 'Error:';
        }
        
        // Scroll to error
        errorContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    /**
     * Collect form data from all inputs
     */
    function collectFormData() {
        const formData = {
            title: document.getElementById('title').value.trim(),
            published_date: document.getElementById('published_date').value,
            journal: document.getElementById('journal').value.trim(),
            doi: document.getElementById('doi').value.trim(),
            abstract: document.getElementById('abstract').value.trim(),
            type: 'journal-article', // Default type
            
            // Settings
            mode: document.querySelector('input[name="mode"]:checked').value,
            inductive: document.querySelector('input[name="inductive"]:checked').value === 'true',
            observed_ratio: parseFloat(document.getElementById('observed_ratio').value),
            top_k: parseInt(document.getElementById('top_k').value),
            device: document.getElementById('device').value
        };
        
        // Collect authors
        formData.authors = [];
        const authorRows = document.querySelectorAll('.author-row');
        
        authorRows.forEach(row => {
            const inputs = row.querySelectorAll('input');
            const firstName = inputs[0].value.trim();
            const lastName = inputs[1].value.trim();
            
            // Only add non-empty authors
            if (firstName || lastName) {
                formData.authors.push([firstName, lastName, null]);
            }
        });
        
        // Collect references
        formData.references = [];
        const referenceItems = document.querySelectorAll('.reference-item');
        
        referenceItems.forEach(item => {
            const refType = item.querySelector('.reference-type').value;
            const title = item.querySelector('.reference-title').value.trim();
            const year = item.querySelector('.reference-year').value.trim();
            const journal = item.querySelector('.reference-journal').value.trim();
            const doi = item.querySelector('.reference-doi').value.trim();
            
            // Collect reference authors
            const refAuthors = [];
            const authorRows = item.querySelectorAll('.reference-author-row');
            
            authorRows.forEach(row => {
                const inputs = row.querySelectorAll('input');
                const firstName = inputs[0].value.trim();
                const lastName = inputs[1].value.trim();
                
                // Only add non-empty authors
                if (firstName || lastName) {
                    refAuthors.push([firstName, lastName, null]);
                }
            });
            
            // Add reference if it has at least a title
            if (title) {
                formData.references.push({
                    reference_type: refType,
                    title: title,
                    year: year ? parseInt(year) : null,
                    journal: journal || null,
                    doi: doi || null,
                    authors: refAuthors,
                    working_paper_institution: null // Default value
                });
            }
        });
        
        return formData;
    }
    
    /**
     * Display recommendation results
     */
    function displayResults(data) {
        const resultsInfo = document.getElementById('results-info');
        const metricsContainer = document.getElementById('metrics-container');
        const metricsRow = document.getElementById('metrics-row');
        const recommendationsTable = document.getElementById('recommendations-table');
        
        // Clear previous results
        resultsInfo.innerHTML = '';
        metricsRow.innerHTML = '';
        recommendationsTable.innerHTML = '';
        
        // Paper info
        resultsInfo.innerHTML = `
            <h5>${data.paper_title || 'Your Paper'}</h5>
            <p>Paper ID: ${data.paper_id || 'N/A'}</p>
            <p>Existing Citations: ${data.existing_citations || 0}</p>
        `;
        
        // Evaluation metrics (for evaluate mode)
        if (data.mode === 'evaluate' || data.hasOwnProperty('recall@' + data.top_k)) {
            metricsContainer.classList.remove('d-none');
            
            // Add metrics
            const metrics = [
                { label: 'Held Out Citations', value: data.held_out_citations || 0 },
                { label: 'Hits', value: data.hits || 0 },
                { label: `Recall@${data.top_k || 50}`, value: data[`recall@${data.top_k || 50}`] || 0, percentage: true },
                { label: `Precision@${data.top_k || 50}`, value: data[`precision@${data.top_k || 50}`] || 0, percentage: true },
                { label: `NDCG@${data.top_k || 50}`, value: data[`ndcg@${data.top_k || 50}`] || 0, percentage: true }
            ];
            
            metrics.forEach(metric => {
                const metricEl = document.createElement('div');
                metricEl.className = 'col-md-2 col-sm-4 col-6 mb-3';
                metricEl.innerHTML = `
                    <div class="metric-card">
                        <div class="metric-value">${metric.percentage ? (metric.value * 100).toFixed(2) + '%' : metric.value}</div>
                        <div class="metric-label">${metric.label}</div>
                    </div>
                `;
                metricsRow.appendChild(metricEl);
            });
        } else {
            metricsContainer.classList.add('d-none');
        }
        
        // Recommendations
        if (data.recommendations && data.recommendations.length > 0) {
            data.recommendations.forEach(rec => {
                const row = document.createElement('tr');
                
                // Determine status badge
                let statusBadge = '';
                let statusText = '';
                
                if (rec.status === 'ACTUAL_CITATION') {
                    statusBadge = '<span class="status-badge status-actual">✓</span>';
                    statusText = 'Actual citation (held out)';
                } else if (rec.status === 'NOT_CITED') {
                    statusBadge = '<span class="status-badge status-not-cited">×</span>';
                    statusText = 'Not cited';
                } else if (rec.status === 'NEW_RECOMMENDATION') {
                    statusBadge = '<span class="status-badge status-new">N</span>';
                    statusText = 'New recommendation';
                } else if (rec.status === 'ALREADY_CITED') {
                    statusBadge = '<span class="status-badge status-already">C</span>';
                    statusText = 'Already cited';
                }
                
                // Score bar width based on score
                const maxScore = data.recommendations[0].score;
                const scorePercentage = (rec.score / maxScore) * 100;
                
                row.innerHTML = `
                    <td>${rec.rank}</td>
                    <td title="${statusText}">${statusBadge}</td>
                    <td>
                        <strong>${rec.title}</strong>
                        <span class="score-bar" style="width: ${scorePercentage}%"></span>
                    </td>
                    <td>${rec.authors || 'Unknown'}</td>
                    <td>${rec.journal || 'Unknown'}</td>
                    <td>${rec.score.toFixed(4)}</td>
                `;
                
                recommendationsTable.appendChild(row);
            });
        } else {
            recommendationsTable.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center">No recommendations found</td>
                </tr>
            `;
        }
        
        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    /**
     * Format date string (YYYY-MM-DD) from ISO date
     */
    function formatDate(dateString) {
        if (!dateString) return '';
        
        try {
            const date = new Date(dateString);
            return date.toISOString().split('T')[0];
        } catch (e) {
            console.error('Error formatting date:', e);
            return '';
        }
    }
}); 
