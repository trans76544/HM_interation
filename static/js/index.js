let canvas = document.getElementById("canvas1");
let ctx = canvas.getContext('2d');
let painting = false;
let startPoint = {x: undefined, y: undefined};
let EraserEnabled = false;

wh();

function wh() {
    let pageWidth = document.documentElement.clientWidth;
    let pageHeight = document.documentElement.clientHeight;
    canvas.width = pageWidth;
    canvas.height = pageHeight;
}

canvas.onmousedown = function (e) {
    let x = e.offsetX;
    let y = e.offsetY;
    painting = true;
    if (EraserEnabled) {
        ctx.clearRect(x - 15, y - 15, 30, 30)
    }
    startPoint = {x: x, y: y};
};

canvas.onmousemove = function (e) {
    let x = e.offsetX;
    let y = e.offsetY;
    let newPoint = {x: x, y: y};
    if (painting) {
        if (EraserEnabled) {
            ctx.clearRect(x - 15, y - 15, 30, 30)
        } 
		else {
            drawLine(startPoint.x, startPoint.y, newPoint.x, newPoint.y);
        }
        startPoint = newPoint;
    }
};
//    松开鼠标
//    鼠标松开事件（onmouseup)
canvas.onmouseup = function () {
    painting = false;
};

function drawLine(xStart, yStart, xEnd, yEnd) {
    ctx.beginPath();
	ctx.strokeStyle = 'white';
    ctx.lineWidth = 8;
    ctx.moveTo(xStart, yStart);
    ctx.lineTo(xEnd, yEnd);
    ctx.stroke();
    ctx.closePath();
}

eraser.onclick = function () {
    EraserEnabled = true;
    eraser.classList.add('active');
    pen.classList.remove('active');
    canvas.classList.add('xiangpica');
};

pen.onclick = function () {
    EraserEnabled = false;
    pen.classList.add('active');
    eraser.classList.remove('active');
    canvas.classList.remove('xiangpica');
};

clean.onclick = function() {
    ctx.fillStyle = '#293937';
    ctx.fillRect(0,0,canvas.width,canvas.height);
};

download.onclick = function() {
    let url = canvas.toDataURL('image/png');
    let a = document.createElement('a');
    document.body.appendChild(a);
    a.href = url;
    a.download = '草稿纸';
    a.target = '_blank';
    a.click()
};

scan.onclick = function() {
	let url = canvas.toDataURL('image/png');
	var img = new Image();
	img.src = url;
	// var formData = new FormData();
	// formData.append("image", "aa");
	// // formData.append('image', image);
	// console.log(formData["image"])
	// $.ajax({
	//     url         : 'http://127.0.0.1:5000/predict',
	//     data        : formData,
	//     processData : false,
	//     contentType : false,
	//     type: 'POST'
	// }).done(function(data){
	//     console.log(data);
	// });
	fetch(img.src)
	    .then(res => res.blob())
	    .then(blob => {
	        const file = new File([blob], "capture.png", {
	            type: 'image/png'
	        });
	        var fd = new FormData();
	        fd.append("image", file);
	        $.ajax({
	            type: "POST",
	            enctype: 'multipart/form-data',
	            url: "http://127.0.0.1:5000/predict",
	            data: fd,
	            processData: false,
	            contentType: false,
	            cache: false,
	            success: (data) => {
	                console.log(data['prediction']);
					var label = document.getElementById("ans");
					str = "识别结果为";
					str = str.concat(data['prediction'])
					label.innerText = str;
					// alert("识别结果是", data['prediction']);
	            },	
	            error: function(xhr, status, error) {
	                alert(xhr.responseText);
	            }
	        });
	    });
};