// open-close menu
// const icon = document.getElementsByClassName('fa-bars');
let navIcon = document.querySelector('.nav-icon');
let menuIcon = document.querySelector('.menu-icon');
let menu = document.querySelector('.menu');
let menuBefore = document.querySelector(('.menu'), ':before');

let menuContainer = document.querySelector('.menu-container');
let videoContainer = document.querySelector('.video_container');


navIcon.onclick = () => {
    menu.style.display = 'block';
};

menuIcon.onclick = () => {
    menu.style.display = 'none';
};

menuBefore.onclick = () => {
    menu.style.display = 'none';
};


// show list course (đường dẫn dựa vào file html được link vào)
// let videoContainer = document.querySelector('.video_container');
let listCourse = "";
fetch("./static/js/database/courseParent.json")

    .then(function (response) {
        return response.json();
    })

    .then(function (course) {
        // console.log(course)
        listCourse = `<div class="course-title">Course</div>
                        <div class="course-container row">`;

        for (let c of course) {

            listCourse += `
            <div class="course-item col l-3 m-6 c-12" onclick="watch('${c.title}','${c.url}','${c.date}','${c.description}')">
                <a href="#" class="course-item-link">
                <img src="${c.img}" alt=""class="course-item-img">
                </a>
            <h1 class="course-item-title">${c.title}</h1>
            </div>
            `;

        }
        listCourse += `</div>
                    </div > `;
        videoContainer.innerHTML = listCourse;
    })


    .catch(function (error) {
        console.error('Error:', error);
    });


//show video when click
let video = "";
let titlePage = document.querySelector('title');

function watch(title, url, date, description) {
    // let videoContainer = document.querySelector('.video_container');

    console.log(title)
    video = `
        <iframe src="${url + "?rel=0&enablejsapi=1"}"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
        
        <div class="video-title">${title}</div>

        <div class="video-description">

            <div class="video-date">
                ${date}
            </div>

            <div class="video-content">
                ${description}
            </div>

            <a href="${url}" class="video-link">Source:${url}</a>
 
        </div> 
        `;

    titlePage.innerHTML = `${title}`;
    videoContainer.innerHTML = video + listCourse;

    //call check face

}