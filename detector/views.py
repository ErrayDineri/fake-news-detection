from django.shortcuts import render
import re, os, random, torch
from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected
import torch.nn.functional as F

from .models import TweetPrediction
from .gcn_predict import GNNFakeNews

# ─── Helpers ────────────────────────────────────────────────────────────────
def extract_tweet_id(url):
    for pattern in (r'twitter\.com/\w+/status/(\d+)',
                    r'x\.com/\w+/status/(\d+)'):
        m = re.search(pattern, url)
        if m: return m.group(1)
    return None

# ─── Model setup ────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'gnn_fakenews.pt')
)
gnn_model = GNNFakeNews('GCN', in_channels=768, hidden_channels=128, out_channels=2)
gnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
gnn_model.to(DEVICE).eval()

# ─── Load UPFD test split once ───────────────────────────────────────────────
DATA_ROOT     = os.path.join(os.path.dirname(__file__), 'data', 'UPFD')
try:
    test_dataset = UPFD(
        DATA_ROOT, name='gossipcop', feature='bert', split='test',
        transform=ToUndirected()
    )
    # sanity-check:
    fake_count = sum(1 for d in test_dataset if int(d.y) == 0)
    real_count = sum(1 for d in test_dataset if int(d.y) == 1)
    print(f"[UPFD] test split: {len(test_dataset)} graphs → fake={fake_count}, real={real_count}")
except Exception as e:
    print(f"[ERROR] loading UPFD test split: {e}")
    test_dataset = []


# ─── Inference routines ─────────────────────────────────────────────────────
def run_fake_news_detection(tweet_id):
    # your real preprocessing should go here; for demo we keep it minimal
    x = torch.zeros((10, 768), dtype=torch.float, device=DEVICE)
    edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,9],
                               [1,2,3,4,5,6,7,8,9,0]], device=DEVICE)
    batch = torch.zeros(10, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out = gnn_model(x, edge_index, batch)
        return out.argmax(dim=1).item()

def run_fake_news_detection_random():
    if not test_dataset:
        return None, None

    idx  = random.randrange(len(test_dataset))
    data = test_dataset[idx].to(DEVICE)

    with torch.no_grad():
        out   = gnn_model(data.x, data.edge_index, data.batch)
        probs = F.softmax(out, dim=1).squeeze().tolist()
        pred  = int(out.argmax(dim=1).item())
        true  = int(data.y.item())

    print(f"[RANDOM] idx={idx}, true={true}, logits={out.squeeze().tolist()}, probs={probs}")

    return pred, true


# ─── Django view ────────────────────────────────────────────────────────────
def index(request):
    context = {
        'result':     None,
        'tweet_id':   None,
        'true_label': None,
    }

    if request.method == 'POST':
        if 'random_test' in request.POST:
            pred, true = run_fake_news_detection_random()
            context['result']     = str(pred)
            context['tweet_id']   = f"random_idx_{true}"  # or f"random_{idx}" if you want the index
            context['true_label'] = str(true)

        else:
            twitter_url = request.POST.get('twitter_url','').strip()
            tweet_id    = extract_tweet_id(twitter_url)

            if tweet_id:
                try:
                    obj = TweetPrediction.objects.get(tweet_id=tweet_id)
                    result = obj.prediction
                except TweetPrediction.DoesNotExist:
                    result = run_fake_news_detection(tweet_id)
                    if str(result) in ('0','1'):
                        TweetPrediction.objects.create(
                            tweet_id=tweet_id,
                            prediction=str(result)
                        )
                context['result']   = str(result)
                context['tweet_id'] = tweet_id

    return render(request, 'detector/index.html', context)
