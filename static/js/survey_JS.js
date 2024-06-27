document.addEventListener('DOMContentLoaded', () => {
    let Question = document.querySelector('#ques');
    let answerContainer = document.querySelector('#answer-container');
    let answerBtn = document.querySelectorAll('.btn');
    let nextBtn = document.querySelector('.next-btn');
    let countQues = document.querySelector('.number-ques');
    let formContainer = document.querySelector('.form');

    // Check if essential elements exist
    if (!Question || !answerContainer || !nextBtn || !countQues || !formContainer) {
        return;
    }

    let ques = [];

    fetch("./static/js/database/question.json")
        .then(response => response.json())
        .then(question => {
            for (let i = 1; i <= 10; i++) {
                let q = question[random(question.length)].ques;
                ques.push(q);
            }
            return ques;
        })
        .then(arr => {
            nextBtn.addEventListener("click", () => {
                if (currQuesIndex < arr.length) {
                    handleNextBtn(arr);
                }
            });

            answerBtn.forEach(ans => {
                ans.addEventListener("keypress", e => {
                    if (e.key === "Enter") {
                        e.preventDefault();
                        if (currQuesIndex < arr.length) {
                            handleNextBtn(arr);
                        }
                    }
                });
            });

            startQues(arr);
        })
        .catch(error => {
            console.error('Error:', error);
        });

    function random(len) {
        return Math.floor(Math.random() * len);
    }

    let currQuesIndex = 0;
    let score = 0;

    function startQues(arr) {
        currQuesIndex = 0;
        score = 0;
        nextBtn.innerHTML = "Next";
        showQues(arr);
    }

    function showQues(arr) {
        let question = arr[currQuesIndex];

        if (countQues) {
            if (currQuesIndex >= 1) {
                const currWidth = countQues.offsetWidth;
                const newWidth = currWidth * (currQuesIndex + 1);
                countQues.style.width = newWidth + 'px';
                countQues.innerHTML = currQuesIndex + 1;
            }
        }

        let questionNO = currQuesIndex + 1;
        Question.innerHTML = questionNO + ". " + question;

        nextBtn.style.display = "none";

        answerBtn.forEach(btn => btn.classList.remove("add-Color"));
    }

    function showScore() {
        formContainer.style.display = 'block';
        formContainer.classList.add("active");
    }

    function handleNextBtn(arr) {
        currQuesIndex++;
        if (currQuesIndex < arr.length) {
            showQues(arr);
        } else {
            showScore();
        }
    }

    answerBtn.forEach((btn, index) => {
        btn.addEventListener('click', () => {
            btn.classList.add("add-Color");
            answerBtn.forEach(otherBtn => {
                if (otherBtn !== btn) {
                    otherBtn.classList.remove("add-Color");
                }
            });
            nextBtn.style.display = "block";

            switch (index) {
                case 0: score += 0; break;
                case 1: score += 1; break;
                case 2: score += 2; break;
                case 3: score += 5; break;
                case 4: score += 10; break;
            }
        });
    });

    document.querySelector('form').addEventListener('submit', function (event) {
        event.preventDefault();

        let parentName = document.querySelector('.parentName').value;
        let childrenName = document.querySelector('.childrenName').value;

        if (parentName && childrenName) {
            const data = {
                parentName: parentName,
                childrenName: childrenName,
                Score: score,
            };
            postData(data);
        } else {
            alert("Input Information !!");
        }

        formContainer.style.display = 'none';
        formContainer.classList.remove("active");

        answerContainer.style.display = 'none';

        let surveyTitle = document.querySelector('.survey-title');
        surveyTitle.innerHTML = `Thank you for taking the survey!`;
        surveyTitle.style.textAlign = "center";

        Question.style.textAlign = "center";
        if (score <= 30) {
            Question.innerHTML = `Theo chuẩn đoán, con bạn mắc chứng tự kỷ cấp độ 1. Đây chỉ là chuẩn đoán từ những dữ liệu có sẵn, cần liên hệ chuyên gia để biết thêm!`
        } else if (score > 30 && score <= 70) {
            Question.innerHTML = `Theo chuẩn đoán, con bạn mắc chứng tự kỷ cấp độ 2. Đây chỉ là chuẩn đoán từ những dữ liệu có sẵn, cần liên hệ chuyên gia để biết thêm!`
        } else if (score > 70) {
            Question.innerHTML = `Theo chuẩn đoán, con bạn mắc chứng tự kỷ cấp độ 3. Đây chỉ là chuẩn đoán từ những dữ liệu có sẵn, cần liên hệ chuyên gia để biết thêm!`
        }
        let backHomeBtn = document.querySelector('.backHome-btn');
        countQues.style.display = 'none';
        nextBtn.style.display = 'none';
        backHomeBtn.style.display = 'block';
        backHomeBtn.innerHTML = `Trang chủ`;
        backHomeBtn.onclick = () => {
            window.location.href = '/';
        }
    });

    async function postData(data) {
        const formData = new FormData();
        formData.append("entry.1218025536", data.parentName);
        formData.append("entry.1487688909", data.childrenName);
        formData.append("entry.1626381746", data.Score);

        fetch("https://docs.google.com/forms/d/e/1FAIpQLSeOjDxcA7wDFzNd5vjUhJehhsvsQ7s_ix_d_peGhYwTRCLuJg/formResponse", {
            method: "POST",
            body: formData,
            mode: "no-cors",
        });
    }
});
