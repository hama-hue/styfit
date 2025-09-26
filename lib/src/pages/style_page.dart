import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker_web/image_picker_web.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class StylePage extends StatefulWidget {
  const StylePage({super.key});
  @override
  State<StylePage> createState() => _StylePageState();
}

class _StylePageState extends State<StylePage> {
  String? imageUrl;
  List<dynamic>? picks;
  bool loading = false;

  Future<void> pickAndUpload() async {
    setState(() => loading = true);
    final bytes = await ImagePickerWeb.getImageAsBytes();
    if (bytes == null) { setState(()=>loading=false); return; }

    final uid = FirebaseAuth.instance.currentUser!.uid;
    final ref = FirebaseStorage.instance.ref('user_uploads/$uid/${DateTime.now().millisecondsSinceEpoch}.jpg');
    await ref.putData(bytes, SettableMetadata(contentType: 'image/jpeg'));
    final url = await ref.getDownloadURL();
    setState(() => imageUrl = url);

    // call backend Node API
    final token = await FirebaseAuth.instance.currentUser!.getIdToken();
    final resp = await http.post(
      Uri.parse('http://localhost:5000/api/style'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: jsonEncode({'imageUrl': url, 'occasion': 'casual'}),
    );

    if (resp.statusCode == 200) {
      final data = jsonDecode(resp.body);
      setState(() => picks = data['picks'] ?? []);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Error from backend')));
    }
    setState(() => loading = false);
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(children: [
        ElevatedButton.icon(onPressed: pickAndUpload, icon: const Icon(Icons.upload), label: const Text('Upload photo')),
        if (loading) const Padding(padding: EdgeInsets.all(8), child: CircularProgressIndicator()),
        if (imageUrl != null) Image.network(imageUrl!, height: 240),
        const SizedBox(height: 12),
        Expanded(
          child: picks == null ? const Text('No recommendations yet') :
          ListView.builder(
            itemCount: picks!.length,
            itemBuilder: (c,i){
              final p = picks![i];
              return Card(
                child: ListTile(
                  leading: Image.network(p['imageUrl'], width: 64, height: 64, fit: BoxFit.cover),
                  title: Text(p['title'] ?? 'Item'),
                  subtitle: Text('\$${p['price'] ?? '0'} • ${p['brand'] ?? ''}'),
                  trailing: IconButton(icon: const Icon(Icons.open_in_new), onPressed: () => _open(p['buyUrl'])),
                ),
              );
            }
          ),
        )
      ]),
    );
  }

  void _open(String? url) {
    if (url == null) return;
    // open in new tab (web)
    // ignore: unsafe_html
    // use dart:html only on web
    // workaround:
    try {
      // works in web: import 'dart:html' as html; html.window.open(url, '_blank');
    } catch (e) {}
  }
}
