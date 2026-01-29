const dropdownData = {
    THEORY: [
<<<<<<< HEAD
        { label: "T.1: Thomson Scattering", link: "https://en.wikipedia.org/wiki/Thomson_scattering" },
        { label: "T.2: Fluctuation-Dissipation Approach", link: "https://en.wikipedia.org/wiki/Fluctuation-dissipation_theorem" }
    ],
    OBSERVATION: [
        { label: "O.1: Standard Madrigal Outputs", link: "https://en.wikipedia.org/wiki/Haystack_Observatory" },
        { label: "O.2: Instabilities", link: "https://en.wikipedia.org/wiki/Incoherent_scatter" }
    ],
    EXPERIMENT: [
        { label: "E.1: Overview of Existing Radars", link: "https://en.wikipedia.org/wiki/List_of_radars" },
        { label: "E.2: Basic Experimental Setup", link: "#" }
    ],
    JUPYTER_NOTEBOOK: [
        { label: "J.1: Jupyter Testing", link: "https://nbviewer.org/github/DredgenYors/JupyterNotebookTesting/blob/main/JupyterTesting.ipynb" }
=======
        { label: "T.1: Thomson Scattering (Temp. for Testing)", link: "https://en.wikipedia.org/wiki/Thomson_scattering" },
        { label: "T.2: Fluctuation-Dissipation Approach (Temp. for Testing)", link: "https://en.wikipedia.org/wiki/Fluctuation-dissipation_theorem" }
    ],
    OBSERVATION: [
        { label: "O.1: Standard Madrigal Outputs (Temp. for Testing)", link: "https://en.wikipedia.org/wiki/Haystack_Observatory" },
        { label: "O.2: Instabilities (Temp. for Testing)", link: "https://en.wikipedia.org/wiki/Incoherent_scatter" }
    ],
    EXPERIMENT: [
        { label: "E.1: Overview of Existing Radars (Temp. for Testing)", link: "https://en.wikipedia.org/wiki/List_of_radars" },
        { label: "E.2: Basic Experimental Setup (Temp. for Testing)", link: "#" }
    ],
    JUPYTER_NOTEBOOK: [
        { 
            label: "JB.1: Basic Functions for Spectra Calculation", 
            link: "https://nbviewer.org/github/DredgenYors/JupyterNotebookTesting/blob/main/JupyterTesting.ipynb" 
        },
        { 
            label: "JB.2: Spectra Calculation with Multi-Ion and Varying Temperature Support", 
            link: "https://nbviewer.org/github/DredgenYors/Updated-Spectra-Jupyter-Notebook-August-2025/blob/main/ISR_spectra_JB.ipynb" 
        }
>>>>>>> 91364ff5daad93d415acdf0c7845bc9a90f3aa85
    ],
    SIMULATION: [
        { label: "S.1: Simulation Setup", link: "#" },
        { label: "S.2: Results Analysis", link: "#" }
    ]
};

<<<<<<< HEAD
function toggleDropdown(category) {
    const dropdownContainer = document.getElementById("dropdown-container");
    if (dropdownContainer.style.display === "block") {
        dropdownContainer.style.display = "none";
        return;
    }

    dropdownContainer.innerHTML = dropdownData[category]
        .map(item => `<a href="${item.link}" target="_blank" class="dropdown-item">${item.label}</a>`)
        .join("");
    dropdownContainer.style.display = "block";
}

=======
// ================================================================================
// DROPDOWN MENU FUNCTIONALITY
// ================================================================================

/**
 * Toggle dropdown menu visibility and populate with category-specific content.
 * 
 * This function handles the main dropdown menu behavior:
 * 1. Checks current visibility state
 * 2. Clears existing content
 * 3. Either hides menu or populates with new content
 * 4. Creates clickable links that open in new tabs
 * 
 * @param {string} category - Category key from dropdownData object
 */
let currentDropdownCategory = null; // Track the currently open category

function toggleDropdown(category) {
    const dropdownContainer = document.getElementById("dropdown-container");

    // If the same category is clicked, toggle (close) the dropdown
    if (currentDropdownCategory === category && dropdownContainer.style.display === "block") {
        dropdownContainer.style.display = "none";
        currentDropdownCategory = null;
        return;
    }

    // Otherwise, show and repopulate the dropdown with the selected category
    dropdownContainer.innerHTML = "";

    dropdownData[category].forEach(item => {
        const link = document.createElement("a");
        link.href = item.link;
        link.target = "_blank";
        link.classList.add("dropdown-item");
        link.textContent = item.label;
        if (item.tooltip) link.title = item.tooltip;
        dropdownContainer.appendChild(link);
    });

    dropdownContainer.style.display = "block";
    currentDropdownCategory = category;
}

// ================================================================================
// IFRAME CONTENT MANAGEMENT
// ================================================================================

/**
 * Display extra ISR content in an iframe-based interface.
 * 
 * This function provides an alternative content viewing method:
 * 1. Updates the section header with category name
 * 2. Creates a list of clickable items
 * 3. Loads selected content into an iframe for embedded viewing
 * 4. Prevents default link behavior to control iframe navigation
 * 
 * @param {string} category - Category key from dropdownData object
 * 
 * Usage: Typically called when user wants to view content without leaving the page
 */
function showExtraISR(category) {
    // Update section header with formatted category name
    // Replace underscores with spaces for better readability
    document.getElementById("extra-header").textContent = category.replace("_", " ");
    
    // Get the container for extra links
    const extraLinks = document.getElementById("extra-links");
    
    // Clear any existing extra links
    extraLinks.innerHTML = "";

    // Create list items for each category item
    dropdownData[category].forEach(item => {
        // Create list item container
        const li = document.createElement("li");
        
        // Create anchor element
        const link = document.createElement("a");
        link.href = "#";                          // Prevent default navigation
        link.textContent = item.label;           // Set display text
        
        // Add click handler for iframe loading
        link.onclick = (e) => {
            e.preventDefault();                   // Prevent default link behavior
            
            // Load content into iframe
            document.getElementById("extra-iframe").src = item.link;
            
            // Make iframe visible
            document.getElementById("extra-iframe").style.display = "block";
        };
        
        // Assemble the list item
        li.appendChild(link);
        extraLinks.appendChild(li);
    });
}
>>>>>>> 91364ff5daad93d415acdf0c7845bc9a90f3aa85

