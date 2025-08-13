const dropdownData = {
    // Theoretical foundations and mathematical background
    THEORY: [
        { 
            label: "T.1: Thomson Scattering", 
            link: "https://en.wikipedia.org/wiki/Thomson_scattering" 
        },
        { 
            label: "T.2: Fluctuation-Dissipation Approach", 
            link: "https://en.wikipedia.org/wiki/Fluctuation-dissipation_theorem" 
        }
    ],
    
    // Observational techniques and data analysis
    OBSERVATION: [
        { 
            label: "O.1: Standard Madrigal Outputs", 
            link: "https://en.wikipedia.org/wiki/Haystack_Observatory" 
        },
        { 
            label: "O.2: Instabilities", 
            link: "https://en.wikipedia.org/wiki/Incoherent_scatter" 
        }
    ],
    
    // Experimental setups and radar systems
    EXPERIMENT: [
        { 
            label: "E.1: Overview of Existing Radars", 
            link: "https://en.wikipedia.org/wiki/List_of_radars" 
        },
        { 
            label: "E.2: Basic Experimental Setup", 
            link: "#" 
        }
    ],
    
    // Interactive computational demonstrations
    JUPYTER_NOTEBOOK: [
        { 
            label: "J.1: Jupyter Testing", 
            link: "https://nbviewer.org/github/DredgenYors/JupyterNotebookTesting/blob/main/JupyterTesting.ipynb" 
        }
    ],
    
    // Simulation and modeling tools
    SIMULATION: [
        { 
            label: "S.1: Simulation Setup", 
            link: "#" 
        },
        { 
            label: "S.2: Results Analysis", 
            link: "#" 
        }
    ]
};

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
function toggleDropdown(category) {
    // Get the dropdown container element
    const dropdownContainer = document.getElementById("dropdown-container");
    
    // Clear any existing dropdown content
    dropdownContainer.innerHTML = "";

    // Check if dropdown is currently visible
    if (dropdownContainer.style.display === "block") {
        // Hide dropdown if already visible
        dropdownContainer.style.display = "none";
    } else {
        // Show dropdown and populate with category content
        dropdownData[category].forEach(item => {
            // Create anchor element for each menu item
            const link = document.createElement("a");
            link.href = item.link;                    // Set destination URL
            link.target = "_blank";                   // Open in new tab
            link.classList.add("dropdown-item");     // Add CSS class for styling
            link.textContent = item.label;           // Set display text
            
            // Add link to dropdown container
            dropdownContainer.appendChild(link);
        });
        
        // Make dropdown visible
        dropdownContainer.style.display = "block";
    }
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

