# F1 com IA

Carrinhos com redes neurais pequenas aprendem a dirigir por **neuroevolução** (seleção, cruzamento e mutação).
A pista inicial é uma versão **estilizada** de Interlagos (~4,3 km, desenhada à mão; não é GPS real — use `--tracks csv:arquivo.csv` para uma linha central real).

## Arquitetura
- `f1ia/` — **simulação headless** (numpy, sem pygame): `track`, `sim` (física + sensores), `nn`, `evolution`, `autopilot`, `train`, `evaluate`, `compare`, `storage`, `plots`.
- `f1ia/ui/` — janela pygame; apenas lê o estado da simulação.
- `app.py` + `public/` — API FastAPI e visualizador web para a Vercel.

## Uso
```bash
pip install -r requirements-dev.txt
python -m f1ia train --run runs/interlagos --gens 100 --workers 4 --plot   # sem janela
python -m f1ia train --run runs/interlagos --resume --gens 50              # continua
python -m f1ia play --mode manual            # setas/WASD; M alterna p/ piloto automático
python -m f1ia watch --run runs/interlagos --gen 1,15,best --ghosts        # replay / fantasmas ([ ] navega)
python -m f1ia live --run runs/live          # treino com visualização
python -m f1ia compare --run runs/interlagos --gens 1,5,15,best --out cmp.png
python -m f1ia evaluate --run runs/interlagos --tracks seed:100-119        # generalização
python -m f1ia track --tracks seed:1-4       # PNG de pistas por semente
python -m f1ia plot --run runs/interlagos
python -m f1ia export-web --run runs/interlagos   # atualiza public/data e models/
python dev.py                                # site local em http://localhost:8000
pytest
```
Treinar em várias pistas: `--tracks interlagos,seed:1-5` (fitness = média). `--val-tracks` mede a generalização a cada N gerações.

## Como funciona
- **Carro**: modelo de bicicleta simplificado, volante com taxa limitada, aderência lateral máxima (acima dela o carro "abre" a curva).
- **Rede**: 9 entradas (7 raios, velocidade, volante) → 12 → 8 → 2 (direção, acelerador/freio), 242 pesos; o genoma é um vetor plano.
- **Evolução**: elitismo (4), torneio, cruzamento uniforme/aritmético, mutação gaussiana com decaimento (e aumento se estagnar), 3 imigrantes aleatórios por geração.
- **Fitness**: fração da volta + bônus por completar e por tempo (1,0 = volta completa).
- **Checkpoints**: `runs/<nome>/checkpoints/gen_NNNN.npz` (população ordenada) + `history.csv`. Replays são re-simulações determinísticas do melhor genoma de cada geração.

## Resultado (treino real, 100 gerações, só em Interlagos)
Geração 1 sai da pista; geração 15 completa em 128,4 s; geração 100 completa em 115,1 s (≈ piloto automático). Gráficos em `docs/evolucao.png` e `docs/comparacao.png`.

## Vercel
`app.py` é detectado automaticamente (FastAPI); `public/` vai para a CDN; `requirements.txt` tem só fastapi e numpy. Rotas: `/api/health`, `/api/models`, `/api/track`, `/api/simulate?track=seed:7&model=interlagos_best`. Deploy: `vercel deploy`.
