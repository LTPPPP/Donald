document.addEventListener('DOMContentLoaded', () => {
    let navIcon = document.querySelector('.nav-icon');
    let menuIcon = document.querySelector('.menu-icon');
    let menu = document.querySelector('.menu');
    let videoContainer = document.querySelector('.video_container');

    navIcon.onclick = () => {
        menu.style.display = 'block';
    };

    menuIcon.onclick = () => {
        menu.style.display = 'none';
    };

    window.onclick = (event) => {
        if (event.target === menu) {
            menu.style.display = 'none';
        }
    };

    let listCourse = "";
    fetch("./static/js/database/courseChild.json")
        .then(response => response.json())
        .then(course => {
            listCourse = `<div class="course-title">Course</div>
                            <div class="course-container row">`;
            for (let c of course) {
                listCourse += `
                <div class="course-item col l-3 m-6 c-12" onclick="watch('${c.title}','${c.url}','${c.date}','${c.description}')">
                    <a href="#" class="course-item-link">
                    <img src="${c.img}" alt="" class="course-item-img">
                    </a>
                <h1 class="course-item-title">${c.title}</h1>
                </div>
                `;
            }
            listCourse += `</div></div>`;
            videoContainer.innerHTML = listCourse;
        })
        .catch(error => {
            console.error('Error:', error);
        });

    let video = "";
    let titlePage = document.querySelector('title');

    window.watch = function (title, url, date, description) {
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
                    ${description}<br><br>
                    Chào mừng các bạn đến với kênh chính chủ của dự án Donald với những thước phim, câu chuyện giúp hỗ trợ tăng mức độ tập trung cho trẻ em.
                    Hãy ủng hộ dự án chúng mình bằng cách like, nhận xét và đừng quên đăng ký nhé!
                </div>
                <a href="${url}" class="video-link">Source: ${url}</a>
            </div> 
            `;
        titlePage.innerHTML = `${title}`;
        videoContainer.innerHTML = video + listCourse;

        loadFaceAPI().then(getCameraStream);
    }

    const loadFaceAPI = async () => {
        await faceapi.nets.faceLandmark68Net.loadFromUri('./static/js/For_Check-Face/model');
        await faceapi.nets.faceRecognitionNet.loadFromUri('./static/js/For_Check-Face/model');
        await faceapi.nets.tinyFaceDetector.loadFromUri('./static/js/For_Check-Face/model');
        await faceapi.nets.faceExpressionNet.loadFromUri('./static/js/For_Check-Face/model');
    }

    function getCameraStream() {
        let checkface = document.getElementById('videoCheckFace');
        let botNotification = document.querySelector('.botNotification');

        if (checkface) {
            if (navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ video: {} })
                    .then(stream => {
                        checkface.srcObject = stream;
                    })
                    .catch(error => {
                        console.error('Error accessing the camera:', error);
                    });
            }

            checkface.addEventListener('playing', () => {
                setInterval(async () => {
                    const detectFace = await faceapi.detectAllFaces(checkface, new faceapi.TinyFaceDetectorOptions());
                    var iframe = document.getElementsByTagName("iframe")[0].contentWindow;
                    if (detectFace.length === 0) {
                        botNotification.style.display = 'block';
                        botNotification.innerHTML = 'Warning: face has left the screen';
                        iframe.postMessage(
                            '{"event":"command","func":"pauseVideo","args":""}',
                            "*"
                        );
                    } else {
                        botNotification.style.display = 'none';
                        iframe.postMessage(
                            '{"event":"command","func":"playVideo","args":""}',
                            "*"
                        );
                    }
                }, 3000);
            });
        } else {
            console.error("Element with id 'videoCheckFace' not found.");
        }
    }
});
