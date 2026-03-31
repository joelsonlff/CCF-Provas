// Vercel Serverless Function — proxy para API do Gemini
// A chave fica em variável de ambiente, nunca exposta ao navegador.

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Método não permitido' });
  }

  const keys = [
    process.env.GEMINI_KEY_1,
    process.env.GEMINI_KEY_2,
    process.env.GEMINI_KEY_3,
    process.env.Google01,
    process.env.Google02,
    process.env.Google03,
  ].filter(Boolean);

  if (keys.length === 0) {
    return res.status(500).json({ error: 'Nenhuma chave de API configurada no servidor.' });
  }

  const body = req.body;

  for (let i = 0; i < keys.length; i++) {
    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${keys[i]}`;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 55000);
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      const data = await response.json();
      if (data.error && (data.error.code === 429 || data.error.code === 503)) {
        continue;
      }
      return res.status(response.status).json(data);
    } catch (err) {
      if (i === keys.length - 1) {
        return res.status(502).json({ error: 'Falha na comunicação com a API do Gemini.' });
      }
    }
  }

  return res.status(429).json({ error: 'Limite de requisições atingido em todas as chaves.' });
}
