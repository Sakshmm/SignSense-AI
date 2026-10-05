/* =========================================================
   SIGN SENSE AI
   FRONTEND CONTROLLER
   Cross-platform: macOS + Windows
========================================================= */


/* =========================================================
   DOM ELEMENTS
========================================================= */

const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");

const clearHistory = document.getElementById("clearHistory");
const historyList = document.getElementById("historyList");

const gestureName = document.getElementById("gestureName");
const patientMessage = document.getElementById("patientMessage");

const confidence = document.getElementById("confidence");
const confidenceText = document.getElementById("confidenceText");
const confidenceFill = document.getElementById("confidenceFill");

const handsDetected = document.getElementById("handsDetected");
const systemHandsDetected =
    document.getElementById("systemHandsDetected");

const voiceText = document.getElementById("voiceText");
const voiceStatus = document.getElementById("voiceStatus");

const emergencyBanner =
    document.getElementById("emergencyBanner");


/* =========================================================
   NAVIGATION
========================================================= */

const mainNavigation =
    document.getElementById("mainNavigation");

const dashboardPage =
    document.getElementById("dashboardPage");

const gestureGuidePage =
    document.getElementById("gestureGuidePage");

const communicationPage =
    document.getElementById("communicationPage");


/* =========================================================
   STATE
========================================================= */

let history = [];
let lastHistoryGesture = null;

/*
   This value is updated directly from MediaPipe landmarks.
   Therefore the UI can show 2/2 Hands even if the detection
   message arrives one frame later.
*/
let liveHandCount = 0;


/* =========================================================
   THEME
========================================================= */

function updateThemeButton() {

    if (!themeToggle || !themeIcon) {
        return;
    }

    const isLight =
        document.body.classList.contains("light");

    themeIcon.textContent =
        isLight ? "☀" : "☾";

    if (themeToggle.lastElementChild) {
        themeToggle.lastElementChild.textContent =
            isLight ? "Light Mode" : "Dark Mode";
    }
}


if (themeToggle) {

    themeToggle.addEventListener(
        "click",
        function () {

            document.body.classList.toggle("light");

            const theme =
                document.body.classList.contains("light")
                    ? "light"
                    : "dark";

            localStorage.setItem(
                "signsense-theme",
                theme
            );

            updateThemeButton();
        }
    );
}


const savedTheme =
    localStorage.getItem("signsense-theme");

if (savedTheme === "light") {
    document.body.classList.add("light");
}

updateThemeButton();


/* =========================================================
   SAFE HTML
========================================================= */

function escapeHTML(value) {

    const div = document.createElement("div");

    div.textContent =
        value === null || value === undefined
            ? ""
            : String(value);

    return div.innerHTML;
}


/* =========================================================
   HISTORY
========================================================= */

function addHistory(
    gesture,
    message,
    confidenceValue
) {

    const now = new Date();

    const time =
        now.toLocaleTimeString(
            [],
            {
                hour: "2-digit",
                minute: "2-digit"
            }
        );

    history.unshift({
        gesture: gesture,
        message: message,
        confidence: confidenceValue,
        time: time
    });

    history = history.slice(0, 10);

    renderHistory();
    renderCommunicationHistory();
}


/* =========================================================
   DASHBOARD HISTORY
========================================================= */

function renderHistory() {

    if (!historyList) {
        return;
    }

    if (history.length === 0) {

        historyList.innerHTML =
            '<div class="empty-history">No communication yet</div>';

        return;
    }

    historyList.innerHTML = "";

    history.forEach(function (item) {

        const row =
            document.createElement("div");

        row.className = "history-item";

        row.innerHTML = `
            <div>
                <div class="history-message">
                    ${escapeHTML(item.message)}
                </div>

                <div class="history-meta">
                    ${escapeHTML(item.time)}
                    ·
                    ${escapeHTML(item.gesture)}
                </div>
            </div>

            <div class="history-confidence">
                ${Number(item.confidence).toFixed(1)}%
            </div>
        `;

        historyList.appendChild(row);
    });
}


/* =========================================================
   COMMUNICATION HISTORY
========================================================= */

function renderCommunicationHistory() {

    const list =
        document.getElementById(
            "communicationHistoryList"
        );

    if (!list) {
        return;
    }

    if (history.length === 0) {

        list.innerHTML =
            '<div class="empty-history">No communication yet</div>';

        return;
    }

    list.innerHTML = "";

    history.forEach(function (item) {

        const row =
            document.createElement("div");

        row.className = "history-item";

        row.innerHTML = `
            <div>
                <div class="history-message">
                    ${escapeHTML(item.message)}
                </div>

                <div class="history-meta">
                    ${escapeHTML(item.time)}
                    ·
                    ${escapeHTML(item.gesture)}
                </div>
            </div>

            <div class="history-confidence">
                ${Number(item.confidence).toFixed(1)}%
            </div>
        `;

        list.appendChild(row);
    });
}


/* =========================================================
   CLEAR DASHBOARD HISTORY
========================================================= */

if (clearHistory) {

    clearHistory.addEventListener(
        "click",
        function () {

            history = [];
            lastHistoryGesture = null;

            renderHistory();
            renderCommunicationHistory();
        }
    );
}


/* =========================================================
   CLEAR COMMUNICATION HISTORY
========================================================= */

const communicationClearHistory =
    document.getElementById(
        "communicationClearHistory"
    );

if (communicationClearHistory) {

    communicationClearHistory.addEventListener(
        "click",
        function () {

            history = [];
            lastHistoryGesture = null;

            renderHistory();
            renderCommunicationHistory();
        }
    );
}


/* =========================================================
   HAND COUNT
========================================================= */

function updateHandCountUI(count) {

    let numericCount =
        Number(count);

    if (!Number.isFinite(numericCount)) {
        numericCount = 0;
    }

    numericCount =
        Math.max(
            0,
            Math.min(
                2,
                Math.round(numericCount)
            )
        );

    liveHandCount = numericCount;

    /*
       Main dashboard hand counter
    */

    if (handsDetected) {

        if (numericCount === 0) {

            handsDetected.textContent =
                "0 Hands";

        } else if (numericCount === 1) {

            handsDetected.textContent =
                "1 Hand";

        } else {

            handsDetected.textContent =
                "2 Hands";
        }
    }


    /*
       System Information card
       Example:
       0 / 2 Hands
       1 / 2 Hands
       2 / 2 Hands
    */

    if (systemHandsDetected) {

        systemHandsDetected.textContent =
            `${numericCount} / 2 Hands`;
    }
}


/* =========================================================
   DETECTION UPDATE
========================================================= */

function updateDetection(
    gesture,
    message,
    confidenceValue,
    handCount = 0
) {

    /* -----------------------------------------
       Gesture
    ----------------------------------------- */

    if (gestureName) {

        gestureName.textContent =
            gesture || "Waiting...";
    }


    /* -----------------------------------------
       Patient Message
    ----------------------------------------- */

    if (patientMessage) {

        patientMessage.textContent =
            message || "Waiting for a gesture...";
    }


    /* -----------------------------------------
       Confidence
    ----------------------------------------- */

    let percentage =
        Number(confidenceValue);

    if (!Number.isFinite(percentage)) {
        percentage = 0;
    }

    percentage =
        Math.max(
            0,
            Math.min(
                100,
                percentage
            )
        );


    if (confidence) {

        confidence.textContent =
            `${percentage.toFixed(1)}%`;
    }


    if (confidenceText) {

        confidenceText.textContent =
            `${percentage.toFixed(1)}%`;
    }


    if (confidenceFill) {

        confidenceFill.style.width =
            `${percentage}%`;

        if (percentage >= 80) {

            confidenceFill.style.background =
                "var(--success)";

        } else if (percentage >= 60) {

            confidenceFill.style.background =
                "var(--warning)";

        } else {

            confidenceFill.style.background =
                "var(--danger)";
        }
    }


    /* -----------------------------------------
       HAND COUNT FIX
    -----------------------------------------

       Backend sends handCount.

       MediaPipe landmark stream also updates
       liveHandCount.

       We use the larger value so that if:

           Backend = 1
           MediaPipe = 2

       UI becomes:

           2 / 2 Hands

       instead of getting stuck at 1 / 2 Hands.
    */

    let backendHandCount =
        Number(handCount);

    if (!Number.isFinite(backendHandCount)) {
        backendHandCount = 0;
    }

    backendHandCount =
        Math.max(
            0,
            Math.min(
                2,
                Math.round(backendHandCount)
            )
        );


    const effectiveHandCount =
        Math.max(
            backendHandCount,
            liveHandCount
        );


    updateHandCountUI(
        effectiveHandCount
    );


    /* -----------------------------------------
       HISTORY
    ----------------------------------------- */

    if (
        gesture &&
        gesture !== "-" &&
        gesture !== "Waiting..." &&
        message &&
        percentage >= 60 &&
        gesture !== lastHistoryGesture
    ) {

        addHistory(
            gesture,
            message,
            percentage
        );

        lastHistoryGesture =
            gesture;
    }


    /*
       Reset history lock when hands disappear.
    */

    if (
        !gesture ||
        gesture === "-" ||
        backendHandCount === 0
    ) {

        lastHistoryGesture = null;
    }
}


/* =========================================================
   VOICE
========================================================= */

function setVoiceStatus(status) {

    if (!voiceText || !voiceStatus) {
        return;
    }

    if (status === "speaking") {

        voiceText.textContent =
            "Speaking...";

        voiceStatus.classList.add(
            "speaking"
        );

    } else {

        voiceText.textContent =
            "Voice Ready";

        voiceStatus.classList.remove(
            "speaking"
        );
    }
}


/* =========================================================
   EMERGENCY
========================================================= */

function showEmergency(show = true) {

    if (!emergencyBanner) {
        return;
    }

    if (show) {

        emergencyBanner.classList.add(
            "show"
        );

    } else {

        emergencyBanner.classList.remove(
            "show"
        );
    }
}


/* =========================================================
   LANDMARK DRAWING
========================================================= */

function drawLandmarks(hands) {

    const canvas =
        document.getElementById(
            "landmarkCanvas"
        );

    const cameraFeed =
        document.getElementById(
            "cameraFeed"
        );


    if (!canvas || !cameraFeed) {
        return;
    }


    /*
       IMPORTANT:
       Update hand count directly from MediaPipe.

       This is the second layer of the 2-hand fix.
    */

    const detectedHandCount =
        Array.isArray(hands)
            ? Math.min(2, hands.length)
            : 0;

    updateHandCountUI(
        detectedHandCount
    );


    const rect =
        cameraFeed.getBoundingClientRect();


    if (
        rect.width <= 0 ||
        rect.height <= 0
    ) {
        return;
    }


    canvas.width =
        rect.width;

    canvas.height =
        rect.height;


    const ctx =
        canvas.getContext("2d");


    if (!ctx) {
        return;
    }


    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    if (
        !Array.isArray(hands) ||
        hands.length === 0
    ) {
        return;
    }


    const connections = [

        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],

        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],

        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],

        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],

        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],

        [5, 9],
        [9, 13],
        [13, 17]
    ];


    hands.forEach(function (hand) {

        if (
            !hand ||
            !Array.isArray(hand.points) ||
            hand.points.length === 0
        ) {
            return;
        }


        const points =
            hand.points;


        const coordinates =
            points.map(function (point) {

                return {

                    x:
                        Number(point.x) *
                        canvas.width,

                    y:
                        Number(point.y) *
                        canvas.height
                };
            });


        const xs =
            coordinates.map(
                point => point.x
            );

        const ys =
            coordinates.map(
                point => point.y
            );


        const minX =
            Math.min(...xs);

        const maxX =
            Math.max(...xs);

        const minY =
            Math.min(...ys);

        const maxY =
            Math.max(...ys);


        const padding = 14;


        /*
           Bounding Box
        */

        ctx.beginPath();

        ctx.rect(
            minX - padding,
            minY - padding,
            (maxX - minX) +
                padding * 2,
            (maxY - minY) +
                padding * 2
        );

        ctx.lineWidth = 2;
        ctx.strokeStyle = "#35d07f";
        ctx.stroke();


        /*
           Hand Connections
        */

        ctx.beginPath();

        connections.forEach(
            function (connection) {

                const first =
                    coordinates[
                        connection[0]
                    ];

                const second =
                    coordinates[
                        connection[1]
                    ];


                if (!first || !second) {
                    return;
                }


                ctx.moveTo(
                    first.x,
                    first.y
                );

                ctx.lineTo(
                    second.x,
                    second.y
                );
            }
        );


        ctx.lineWidth = 2.5;
        ctx.strokeStyle = "#35d07f";
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.stroke();


        /*
           Landmark Points
        */

        coordinates.forEach(
            function (point) {

                ctx.beginPath();

                ctx.arc(
                    point.x,
                    point.y,
                    5,
                    0,
                    Math.PI * 2
                );

                ctx.fillStyle = "#ffffff";
                ctx.fill();

                ctx.lineWidth = 2;
                ctx.strokeStyle = "#35d07f";
                ctx.stroke();


                ctx.beginPath();

                ctx.arc(
                    point.x,
                    point.y,
                    2,
                    0,
                    Math.PI * 2
                );

                ctx.fillStyle = "#35d07f";
                ctx.fill();
            }
        );
    });
}


/* =========================================================
   CLEAR LANDMARKS
========================================================= */

function clearLandmarks() {

    const canvas =
        document.getElementById(
            "landmarkCanvas"
        );

    if (!canvas) {
        return;
    }

    const ctx =
        canvas.getContext("2d");

    if (!ctx) {
        return;
    }

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    updateHandCountUI(0);
}


/* =========================================================
   PAGE NAVIGATION
========================================================= */

function showPage(pageName) {

    /*
       Hide every page first.
    */

    if (dashboardPage) {
        dashboardPage.style.display =
            "none";
    }

    if (gestureGuidePage) {
        gestureGuidePage.style.display =
            "none";
    }

    if (communicationPage) {
        communicationPage.style.display =
            "none";
    }


    /*
       Show selected page.
    */

    if (pageName === "dashboard") {

        if (dashboardPage) {
            dashboardPage.style.display =
                "";
        }

        if (emergencyBanner) {
            emergencyBanner.style.display =
                "";
        }

    } else if (
        pageName === "gesture-guide"
    ) {

        if (gestureGuidePage) {
            gestureGuidePage.style.display =
                "block";
        }

        if (emergencyBanner) {
            emergencyBanner.style.display =
                "none";
        }

    } else if (
        pageName === "communication"
    ) {

        if (communicationPage) {
            communicationPage.style.display =
                "block";
        }

        if (emergencyBanner) {
            emergencyBanner.style.display =
                "none";
        }

        renderCommunicationHistory();
    }


    /*
       Active navigation item.
    */

    if (mainNavigation) {

        const navItems =
            mainNavigation.querySelectorAll(
                ".nav-item"
            );


        navItems.forEach(
            function (item) {

                const itemPage =
                    item.getAttribute(
                        "data-page"
                    );


                if (
                    itemPage === pageName
                ) {

                    item.classList.add(
                        "active"
                    );

                } else {

                    item.classList.remove(
                        "active"
                    );
                }
            }
        );
    }
}


/* =========================================================
   NAVIGATION CLICK HANDLERS
========================================================= */

if (mainNavigation) {

    const navItems =
        mainNavigation.querySelectorAll(
            ".nav-item"
        );


    navItems.forEach(
        function (item) {

            item.addEventListener(
                "click",
                function () {

                    const pageName =
                        item.getAttribute(
                            "data-page"
                        );


                    if (!pageName) {
                        return;
                    }


                    showPage(
                        pageName
                    );
                }
            );
        }
    );
}


/* =========================================================
   INITIAL STATE
========================================================= */

updateHandCountUI(0);

updateDetection(
    "Waiting...",
    "Waiting for a gesture...",
    0,
    0
);

setVoiceStatus("ready");

renderHistory();

renderCommunicationHistory();

showPage("dashboard");


/* =========================================================
   PYTHON ↔ JAVASCRIPT BRIDGE
   IMPORTANT:
   METHOD NAMES MUST NOT CHANGE.
========================================================= */

window.signsense = {

    /* -----------------------------------------
       CAMERA
    ----------------------------------------- */

    updateCamera:
        function (imageData) {

            const cameraFeed =
                document.getElementById(
                    "cameraFeed"
                );


            if (cameraFeed) {

                cameraFeed.src =
                    "data:image/jpeg;base64," +
                    imageData;
            }
        },


    /* -----------------------------------------
       DETECTION
    ----------------------------------------- */

    updateDetection:
        function (
            gesture,
            message,
            confidenceValue,
            handCount
        ) {

            updateDetection(

                gesture,

                message,

                confidenceValue * 100,

                handCount
            );
        },


    /* -----------------------------------------
       LANDMARKS
    ----------------------------------------- */

    updateLandmarks:
        function (hands) {

            drawLandmarks(
                hands
            );
        },


    /* -----------------------------------------
       VOICE SPEAKING
    ----------------------------------------- */

    voiceSpeaking:
        function () {

            setVoiceStatus(
                "speaking"
            );
        },


    /* -----------------------------------------
       VOICE READY
    ----------------------------------------- */

    voiceReady:
        function () {

            setVoiceStatus(
                "ready"
            );
        },


    /* -----------------------------------------
       EMERGENCY
    ----------------------------------------- */

    showEmergency:
        function (show) {

            showEmergency(
                show
            );
        }
};