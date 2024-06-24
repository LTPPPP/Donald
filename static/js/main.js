
// open nav menu
let iconMenu = document.querySelector('.fa-bars');
let navMenu = document.querySelector('.nav-menu');

iconMenu.onclick = function () {
    console.lo
    if (navMenu.style.display == 'none') {
        iconMenu.classList.add('fa-x');
        navMenu.style.display = 'block';
    } else {
        iconMenu.classList.remove('fa-x');
        navMenu.style.display = 'none';
    }

}

//slider
let sliderList = document.querySelector('.slider-list');
let sliderItem = document.querySelectorAll('.slider-item');
let sliderDot = document.querySelectorAll('.slider-dot li');
let prev = document.getElementById('button_pre');
let next = document.getElementById('button_next');

let active = 0;
let lengthItems = sliderItem.length - 1;

// next btn
next.onclick = function () {
    if (active + 1 > lengthItems) {
        active = 0;
    } else {
        active++;
    }
    reloadSlider();
}

// prev btn
prev.onclick = function () {
    if (active - 1 < 0) {
        active = lengthItems;
    } else {
        active--;
    }
    reloadSlider();
}

// auto chuyển slider
let refeshSlider = setInterval(() => { next.click() }, 5000);

function reloadSlider() {
    let checkLeft = sliderItem[active].offsetLeft;
    sliderList.style.left = -checkLeft + 'px';

    //lấy dot đang được active (xoá và trở lại bth)
    let lastActiveDot = document.querySelector('.dot-item__active');
    lastActiveDot.classList.remove('dot-item__active');
    lastActiveDot.classList.add('dot-item');

    // lấy dot kế tiếp xoá và thêm active
    sliderDot[active].classList.remove('dot-item');
    sliderDot[active].classList.add('dot-item__active');

    //xoá thời gian đếm auto chuyển slider và gọi lại để đếm lại từ đầu
    clearInterval(refeshSlider);
    refeshSlider = setInterval(() => { next.click() }, 5000)
}

sliderDot.forEach((li, key) => {
    li.addEventListener('click', function () {
        active = key;
        reloadSlider();
    })
})
