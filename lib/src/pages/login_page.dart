import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class LoginPage extends StatelessWidget {
  const LoginPage({super.key});
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: ElevatedButton.icon(
          icon: const Icon(Icons.login),
          label: const Text('Sign in anonymously (quick)'),
          onPressed: () async {
            await FirebaseAuth.instance.signInAnonymously();
          },
        ),
      ),
    );
  }
}
