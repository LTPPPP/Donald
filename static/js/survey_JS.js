// Taking survey
let Question = document.querySelector('#ques');
let answerContainer = document.querySelector('#answer-container');
let answerBtn = document.querySelectorAll('.btn');
let nextBtn = document.querySelector('.next-btn');


let ques = [];


fetch("./static/js/database/question.json")

    .then(function (response) {
        return response.json();
    })

    // Get ramdom 10 Question
    .then(function (question) {
        for (var i = 1; i <= 10; i++) {
            let q = question[random(question.length)].ques;
            ques.push(q);
        }
        return ques;
    })

    //Question handle
    .then(function (arr) {

        //Next Question 
        nextBtn.addEventListener("click", () => {
            if (currQuesIndex < arr.length) {
                handleNextBtn(arr);
            }
        })
        startQues(arr);
    })

    .catch(function (error) {
        console.error('Error:', error);
    });


//Random
function random(len) {
    return randomNumber = Math.floor(Math.random() * len);

}


//Question handle
let currQuesIndex = 0;
let score = 0;


function startQues(arr) {
    currQuesIndex = 0;
    score = 0;
    nextBtn.innerHTML = "Next";
    showQues(arr);
}

function showQues(arr) {
    let question = arr[currQuesIndex]
    let questionNO = currQuesIndex + 1;
    Question.innerHTML = questionNO + ". " + question;


    nextBtn.style.display = "none";

    //remove color for next answer
    answerBtn[0].classList.remove("add-Color");
    answerBtn[1].classList.remove("add-Color");
    answerBtn[2].classList.remove("add-Color");
    answerBtn[3].classList.remove("add-Color");
    answerBtn[4].classList.remove("add-Color");

}

let formContainer = document.querySelector('.form');

function showScore(arr) {
    formContainer.style.display = 'block';
    formContainer.classList.add("active");
}

function handleNextBtn(arr) {
    currQuesIndex++;
    if (currQuesIndex < arr.length) {
        console.log(score)
        showQues(arr);
    } else {
        showScore(arr);
    }
}


//Score
answerBtn[0].onclick = function () {
    answerBtn[0].classList.add("add-Color");
    for (otherBtn of answerBtn) {
        if (otherBtn != answerBtn[0]) {
            otherBtn.classList.remove("add-Color");
        }
    }
    nextBtn.style.display = "block";

    score += 0;

}
answerBtn[1].onclick = function () {
    answerBtn[1].classList.add("add-Color");
    for (otherBtn of answerBtn) {
        if (otherBtn != answerBtn[1]) {
            otherBtn.classList.remove("add-Color");
        }
    }
    nextBtn.style.display = "block";
    score += 1;

}

answerBtn[2].onclick = function () {
    answerBtn[2].classList.add("add-Color");
    for (otherBtn of answerBtn) {
        if (otherBtn != answerBtn[2]) {
            otherBtn.classList.remove("add-Color");
        }
    }
    nextBtn.style.display = "block";
    score += 2;

}
answerBtn[3].onclick = function () {
    answerBtn[3].classList.add("add-Color");
    for (otherBtn of answerBtn) {
        if (otherBtn != answerBtn[3]) {
            otherBtn.classList.remove("add-Color");
        }
    }
    nextBtn.style.display = "block";
    score += 5;

}
answerBtn[4].onclick = function () {
    answerBtn[4].classList.add("add-Color");
    for (otherBtn of answerBtn) {
        if (otherBtn != answerBtn[4]) {
            otherBtn.classList.remove("add-Color");
        }
    }
    nextBtn.style.display = "block";
    score += 10;

}


// const fs = require('fs');

// Taking information
document.querySelector('form').addEventListener('submit', function (event) {
    event.preventDefault();

    // Get input values
    let parentName = document.querySelector('.parentName').value;
    let childrenName = document.querySelector('.childrenName').value;

    // Create JSON object
    if (parentName && childrenName) {
        const data =
        {
            parentName: parentName,
            childrenName: childrenName,
            Score: score,
        };
        postData(data);
    } else {
        alert("Input Information !!");
    }




    //Hide get information 
    formContainer.style.display = 'none';
    formContainer.classList.remove("active");

    answerContainer.style.display = 'none';

    //Show result
    let surveyTitle = document.querySelector('.survey-title');
    surveyTitle.innerHTML = `Cảm ơn bạn đã tham gia cuộc khảo sát!`;
    surveyTitle.style.textAlign = "center"

    Question.style.textAlign = "center"
    if (score <= 30) {
        Question.innerHTML = `Theo đánh giá, con bạn mắc chứng tự kỷ độ 1.`
    } else if (score > 30 && score <= 70) {
        Question.innerHTML = `Theo đánh giá, con bạn mắc chứng tự kỷ độ 2.`
    } else if (score > 70) {
        Question.innerHTML = `Theo đánh giá, con bạn mắc chứng tự kỷ độ 3.`
    }

    nextBtn.style.display = "block";
    nextBtn.innerHTML = `Về Trang Chủ`;
    nextBtn.onclick = () => {
        window.location.href = '/';
    }


});

console.log(nextBtn.style)


// Post data to 
async function postData(data) {
    const formData = new FormData();
    formData.append("entry.1218025536", data.parentName);
    formData.append("entry.1487688909", data.childrenName);
    formData.append("entry.1626381746", data.Score);

    fetch("https://docs.google.com/forms/d/e/1FAIpQLSeOjDxcA7wDFzNd5vjUhJehhsvsQ7s_ix_d_peGhYwTRCLuJg/formResponse", {
        method: "POST",
        body: formData,
        mode: "no-cors",
    })
}


