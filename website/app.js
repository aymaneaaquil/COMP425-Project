
// image paths with corresponding image names
const dictionary = {
    "images/cat1.jpeg": "img1", 
    "images/cat2.jpeg": "img2", 
    "images/cat3.jpeg": "img3"
};

const images = ["images/cat1.jpeg", "images/cat2.jpeg", "images/cat3.jpeg"]
currentImage = ""
resultsArray = []
const name = ""



function showImage(imageEl) {
    const imgElement = document.getElementById("random-image");
    document.getElementById("buttons").style.display = "none";
    imgElement.src = imageEl;

    // Display the image
    imgElement.style.display = "block";

    // Hide image after 2 seconds, then show buttons
    setTimeout(() => {
        imgElement.style.display = "none";
        document.getElementById("buttons").style.display = "flex";
    }, 2000);
}


function userChoice(choice) {
    // hide buttons once choice is selected
    document.getElementById("buttons").style.display = "none";

    const fs = require('fs');

    if (choice === 'R'){
    message = dictionary[currentImage] + " was guessed as real"
    }
    if (choice === 'S'){
        message = dictionary[currentImage] + " was guessed as synthetic"
    }

    resultsArray.push(message);
}

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1)); // Random index
        [array[i], array[j]] = [array[j], array[i]]; // Swap elements
    }
}

// once name is entered, save the name value and open next page
function nameToStart(){
    window.location.href = "testDescription.html"; 
    name = document.getElementById("nameInput").value;
}

// once user clicks "start", open survey page
function startToMain(){
    window.location.href = "main.html";  
}


function main() {
    let copy = [...images];
    shuffleArray(copy);

    function showNextImage() {
        // once all images have been shown, save results and go to closing page
        if (copy.length === 0) {
            window.location.href = "closingMessage.html"; 
            //store results
            console.log(name)
            console.log(resultsArray)
            return;
        }

        currentImage = copy.shift(); // Get next image
        showImage(currentImage);
    }

    window.userChoice = function (choice) {
        document.getElementById("buttons").style.display = "none";

        let message = dictionary[currentImage] + " was guessed as " + (choice === 'R' ? "real" : "synthetic");
        console.log(message); 

        // Proceed to the next image after user choice
        setTimeout(showNextImage, 1000);
    };

    showNextImage();
}


window.onload = main;
