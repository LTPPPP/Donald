// open-close menu
// const icon = document.getElementsByClassName('fa-bars');
let nav = document.querySelector('.menu-icon');

let menu = document.querySelector('.menu');
let menuContainer = document.querySelector('.menu-container');
let videoContainer = document.querySelector('.video_container');

nav.onclick = function () {
    // menuContainer.classList.toggle('l-3', 'l-1');
    if (menu.style.display == 'none') {

        //menu
        menu.style.display = 'block';
        menuContainer.classList.add('l-3');
        menuContainer.classList.remove('l-0');

        //video
        videoContainer.classList.add('l-9');
        videoContainer.classList.remove('l-12');

    } else {
        //menu
        menu.style.display = 'none';
        menuContainer.classList.add('l-0');
        menuContainer.classList.remove('l-3');

        //video
        videoContainer.classList.add('l-12');
        videoContainer.classList.remove('l-9');
    }

};

// //open - close content
// let title = document.querySelectorAll('.menu-title');

// menu.forEach(function (element) {
//     let child = element.childNodes;
//     child.forEach(function (e) {
//         console.log(e.nodeName);
//         if (e.nodeName == "H1") {
//             e.onclick = function () {
//                 child.forEach(function (item) {
//                     if (item.nodeName == "A") {
//                         if (item.style.display == 'none') {
//                             item.style.display = 'block';
//                         } else {
//                             item.style.display = 'none';

//                         }
//                     }
//                 });
//             };
//         }
//     });
//     // child.onclick()

// });


// Check face
let video = document.getElementById('videoCheckFace');
let botNotification = document.querySelector('.botNotification');

//gọi api check khuôn mặt
const loadFaceAPI = async () => {
    await faceapi.nets.faceLandmark68Net.loadFromUri('./static/js/For_Check-Face/model');
    await faceapi.nets.faceRecognitionNet.loadFromUri('./static/js/For_Check-Face/model');
    await faceapi.nets.tinyFaceDetector.loadFromUri('./static/js/For_Check-Face/model');
    // các thông tin dự đoán về biểu cảm trên web cam
    await faceapi.nets.faceExpressionNet.loadFromUri('./static/js/For_Check-Face/model');
}

// Function lấy luồng để mở camera
function getCameraStrem() {
    //nếu trình duyệt có hỗ trợ thì gọi ra để sử dụng
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: {} })
            .then(stream => {
                video.srcObject = stream;
            })
    }
}


video.addEventListener('playing', () => {
    // const canvas = faceapi.createCanvasFromMedia(video);
    // document.body.append(canvas);

    setInterval(async () => {
        const detectFace = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions());
        var iframe = document.getElementsByTagName("iframe")[0].contentWindow;
        // console.log(detectFace);
        if (detectFace.length == 0) {
            // Show notification when dont detect face
            // console.log('Cảnh báo khuông mặt đã rời khỏi khung hình');
            botNotification.style.display = 'block';
            botNotification.innerHTML = 'Warning: face has left the screen';

            //Pause video when dont detect face
            iframe.postMessage(
                '{"event":"command","func":"' + 'pauseVideo' + '","args":""}',
                "*"
            );

        } else {
            botNotification.style.display = 'none';

            //Play video when dont detect face
            iframe.postMessage(
                '{"event":"command","func":"' + 'playVideo' + '","args":""}',
                "*"
            );
        }
    }, 3000)
})

loadFaceAPI().then(getCameraStrem);


