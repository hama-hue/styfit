// Firebase config (replace with your project's config)
const firebaseConfig = {
    apiKey: "AIzaSyCwEV8kfBMVX4wVia4eUq4fH2O-Zw_b_1w",
    authDomain: "my-styfit.firebaseapp.com",
    projectId: "my-styfit",
    messagingSenderId: "511467920913",
    appId: "1:511467920913:web:b580547c0c9d20002e9bb0"
};

firebase.initializeApp(firebaseConfig);

// Fetch Fitness Recommendations
async function getFitness() {
    const user = firebase.auth().currentUser;
    if (!user) {
        alert("Please login first");
        return;
    }

    const idToken = await user.getIdToken(true);
    const goal = document.getElementById("goal").value;
    const level = document.getElementById("level").value;

    // Replace YOUR_BACKEND_URL with deployed backend URL or localhost
    const res = await fetch(`https://YOUR_BACKEND_URL/fitness/recommend?goal=${goal}&level=${level}`, {
        method: "POST",
        headers: { "Authorization": "Bearer " + idToken }
    });

    const data = await res.json();
    const div = document.getElementById("fitnessResults");
    div.innerHTML = "";

    if (!data.recommendations || data.recommendations.length === 0) {
        div.innerHTML = "<p>No recommendations found.</p>";
        return;
    }

    data.recommendations.forEach(ex => {
        div.innerHTML += `
        <div class="rec-card">
            <b>${ex.exercise}</b><br>
            Body Part: ${ex.body_part}<br>
            Level: ${ex.level}<br>
            Equipment: ${ex.equipment}
        </div>`;
    });
}
