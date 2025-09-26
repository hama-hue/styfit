import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'pages/style_page.dart';
import 'pages/fitness_page.dart';
import 'pages/login_page.dart';

class StyFitApp extends StatelessWidget {
  const StyFitApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'StyFit',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const Root(),
    );
  }
}

class Root extends StatefulWidget {
  const Root({super.key});
  @override
  State<Root> createState() => _RootState();
}

class _RootState extends State<Root> {
  int idx = 0;
  final pages = [const StylePage(), const FitnessPage(), const LoginPage()];

  @override
  Widget build(BuildContext context) {
    // If not signed in, show LoginPage
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return const LoginPage();

    return Scaffold(
      appBar: AppBar(title: const Text('StyFit')),
      body: pages[idx],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: idx,
        onTap: (i) => setState(() => idx = i),
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.style), label: 'Style'),
          BottomNavigationBarItem(icon: Icon(Icons.fitness_center), label: 'Fitness'),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Profile'),
        ],
      ),
    );
  }
}
