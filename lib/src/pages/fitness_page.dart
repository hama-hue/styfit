import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

class FitnessPage extends StatefulWidget {
  const FitnessPage({super.key});
  @override
  State<FitnessPage> createState() => _FitnessPageState();
}

class _FitnessPageState extends State<FitnessPage> {
  final _form = GlobalKey<FormState>();
  final ageC = TextEditingController(text: '25');
  final heightC = TextEditingController(text: '170');
  final weightC = TextEditingController(text: '70');
  String goal = 'fat_loss';
  Map? plan;
  bool loading = false;

  Future<void> submit() async {
    if (!_form.currentState!.validate()) return;
    setState(()=>loading=true);
    final token = await FirebaseAuth.instance.currentUser!.getIdToken();
    final resp = await http.post(Uri.parse('http://localhost:5000/api/fitness'),
      headers: {'Content-Type':'application/json','Authorization':'Bearer $token'},
      body: jsonEncode({
        'age': int.parse(ageC.text),
        'height': int.parse(heightC.text),
        'weight': int.parse(weightC.text),
        'goal': goal,
      }),
    );
    if (resp.statusCode == 200) {
      plan = jsonDecode(resp.body)['plan'];
    } else {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Error generating plan')));
    }
    setState(()=>loading=false);
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(children: [
        Form(
          key: _form,
          child: Column(children: [
            TextFormField(controller: ageC, decoration: const InputDecoration(labelText: 'Age'), keyboardType: TextInputType.number),
            TextFormField(controller: heightC, decoration: const InputDecoration(labelText: 'Height (cm)'), keyboardType: TextInputType.number),
            TextFormField(controller: weightC, decoration: const InputDecoration(labelText: 'Weight (kg)'), keyboardType: TextInputType.number),
            DropdownButtonFormField<String>(
              value: goal,
              items: const [
                DropdownMenuItem(value: 'fat_loss', child: Text('Fat loss')),
                DropdownMenuItem(value: 'hypertrophy', child: Text('Muscle gain')),
                DropdownMenuItem(value: 'endurance', child: Text('Endurance')),
              ],
              onChanged: (v) => setState(()=>goal=v!),
              decoration: const InputDecoration(labelText: 'Goal'),
            ),
            const SizedBox(height: 10),
            ElevatedButton(onPressed: submit, child: const Text('Generate Plan')),
            if (loading) const CircularProgressIndicator(),
          ]),
        ),
        const SizedBox(height: 12),
        if (plan != null) Expanded(child: SingleChildScrollView(child: Text(jsonEncode(plan, toEncodable: (o)=>o.toString())))),
      ]),
    );
  }
}
