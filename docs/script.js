// Firebase v9 Modular
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-app.js";
import { getAuth, onAuthStateChanged, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

// Firebase Config
const firebaseConfig = {
    apiKey: "AIzaSyCwEV8kfBMVX4wVia4eUq4fH2O-Zw_b_1w",
    authDomain: "my-styfit.firebaseapp.com",
    projectId: "my-styfit",
    appId: "1:511467920913:web:b580547c0c9d20002e9bb0"
};
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Backend URL
const BACKEND_ORIGIN = "https://styfit-backend.onrender.com"; // replace with your deployed backend

// Elements
const loginBtn = document.getElementById("loginBtn");
const signupBtn = document.getElementById("signupBtn");

// --- Auth State Change ---
onAuthStateChanged(auth, user => {
    if (user) {
        loginBtn.textContent = "Profile";
        loginBtn.onclick = () => openModal("profileModal");
        signupBtn.textContent = "Logout";
        signupBtn.style.background = "#dc3545";
        signupBtn.onclick = doLogout;
        document.getElementById("profileEmail").textContent = `Logged in as: ${user.email}`;
    } else {
        loginBtn.textContent = "Login";
        loginBtn.onclick = () => openModal("loginModal");
        signupBtn.textContent = "Sign Up";
        signupBtn.style.background = "#f72585";
        signupBtn.onclick = () => openModal("signupModal");
    }
});

// --- Auth Functions ---
async function doLogin() {
    const email = document.getElementById("loginEmail").value;
    const pass = document.getElementById("loginPassword").value;
    try {
        await signInWithEmailAndPassword(auth, email, pass);
        closeModal("loginModal");
    } catch (err) { alert("Login failed: " + err.message); }
}

async function doSignup() {
    const email = document.getElementById("signupEmail").value;
    const pass = document.getElementById("signupPassword").value;
    try {
        await createUserWithEmailAndPassword(auth, email, pass);
        closeModal("signupModal");
    } catch (err) { alert("Signup failed: " + err.message); }
}

async function doLogout() {
    await signOut(auth);
    closeModal("profileModal");
}

// --- Modal Helpers ---
function openModal(id) { document.getElementById(id).style.display = "block"; }
function closeModal(id) { document.getElementById(id).style.display = "none"; }

// --- Fitness Fetch ---
async function getFitness() {
    const user = auth.currentUser;
    if (!user) { alert("Please login first"); return; }

    const idToken = await user.getIdToken(true);
    const goal = document.getElementById("goal").value;
    const level = document.getElementById("level").value;

    const busyEl = document.getElementById("busy");
    busyEl.style.display = "block";

    try {
        const res = await fetch(`${BACKEND_ORIGIN}/fitness/recommend?goal=${goal}&level=${level}`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${idToken}` }
        });
        if (!res.ok) throw new Error(`Server error: ${res.status}`);

        const data = await res.json();
        const wrap = document.getElementById("fitnessResults");
        wrap.innerHTML = "";

        if (!data.recommendations || data.recommendations.length === 0) {
            wrap.innerHTML = `<div style='grid-column:1/-1;text-align:center;color:#666'>No exercises found.</div>`;
            return;
        }

        data.recommendations.forEach(ex => {
            const c = document.createElement("div");
            c.className = "rec-card";
            c.innerHTML = `
                <h4>${ex.exercise}</h4>
                <div class="small">Body Part: ${ex.body_part}</div>
                <div class="small">Level: ${ex.level}</div>
                <div class="small">Equipment: ${ex.equipment}</div>
            `;
            wrap.appendChild(c);
        });
    } catch (err) {
        alert("Error fetching exercises: " + err.message);
        console.error(err);
    } finally {
        busyEl.style.display = "none";
    }
}
