const sign_in_btn = document.querySelector("#sign-in-btn");
const sign_up_btn = document.querySelector("#sign-up-btn");
const container = document.querySelector(".container");

// chuyển qua lại login và register
sign_up_btn.addEventListener("click", () => {
    container.classList.add("sign-up-mode");
});

sign_in_btn.addEventListener("click", () => {
    container.classList.remove("sign-up-mode");

});


// login account

let loginForm = document.querySelector('.sign-in-form');
let login = document.querySelector('.btn-submit-login');

login.addEventListener("click", () => {
    let username = document.querySelector('.userName-login').value;
    let password = document.querySelector('.password-login').value;

    if (username == "admin" && password == "admin") {
        //khi nhập đúng chuyển trang
        loginForm.action = "{{ url_for('templates_bp.index') }}";
    } else {

    }

});

signUpButton.addEventListener("click", () => {
    container.classList.add("sign-up-mode");
});

signInButton.addEventListener("click", () => {
    container.classList.remove("sign-up-mode");
});


