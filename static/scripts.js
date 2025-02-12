const dropdownData = {
    THEORY: [
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
    ],
    SIMULATION: [
        { label: "S.1: Simulation Setup", link: "#" },
        { label: "S.2: Results Analysis", link: "#" }
    ]
};

function toggleDropdown(category) {
    const dropdownContainer = document.getElementById("dropdown-container");
    dropdownContainer.innerHTML = "";

    if (dropdownContainer.style.display === "block") {
        dropdownContainer.style.display = "none";
    } else {
        dropdownData[category].forEach(item => {
            const link = document.createElement("a");
            link.href = item.link;
            link.target = "_blank";
            link.classList.add("dropdown-item");
            link.textContent = item.label;
            dropdownContainer.appendChild(link);
        });
        dropdownContainer.style.display = "block";
    }
}

function showExtraISR(category) {
    document.getElementById("extra-header").textContent = category.replace("_", " ");
    const extraLinks = document.getElementById("extra-links");
    extraLinks.innerHTML = "";

    dropdownData[category].forEach(item => {
        const li = document.createElement("li");
        const link = document.createElement("a");
        link.href = "#";
        link.textContent = item.label;
        link.onclick = (e) => {
            e.preventDefault();
            document.getElementById("extra-iframe").src = item.link;
            document.getElementById("extra-iframe").style.display = "block";
        };
        li.appendChild(link);
        extraLinks.appendChild(li);
    });
}
