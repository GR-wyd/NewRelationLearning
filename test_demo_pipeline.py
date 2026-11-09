import io
import math
import json
import tempfile
import urllib.parse
import unittest
from pathlib import Path
from unittest.mock import patch

import demo_pipeline as pipeline


class FakeEncoder:
    def encode(self, concepts):
        vectors = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "gamma": [-1.0, 0.0],
        }
        return {concept: vectors[concept] for concept in concepts}


class FakeQwen:
    def generate(self, prompt, max_new_tokens=None):
        _ = prompt, max_new_tokens
        return '{"targets":["beta", "gamma"]}'


class FakeRelationQwen:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_new_tokens=None):
        _ = max_new_tokens
        self.prompts.append(prompt)
        if "Apply the learned relation pattern" in prompt:
            return '{"concept_pairs":[{"source":"alpha","target":"beta","reason":"shared regulation pattern"}]}'
        return '{"relation_pattern":"X regulates Y through a context-dependent mechanism"}'


class FakeRelationEncoder:
    def encode(self, texts):
        return {text: [1.0, 0.0] for text in texts}


class FakeEvidenceQwen:
    model_path = "fake-qwen"

    def __init__(self, response='{"label":"uncertain","reason":"ambiguous"}', error=None):
        self.response = response
        self.error = error
        self.prompts = []

    def generate(self, prompt, max_new_tokens=None):
        _ = max_new_tokens
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.response


class DemoPipelineTests(unittest.TestCase):
    def test_request_json_accepts_literal_control_characters_in_strings(self):
        response = io.BytesIO(b'{"esearchresult":{"querytranslation":"alpha\tbeta","count":"1"}}')
        with patch.object(pipeline.urllib.request, "urlopen", return_value=response):
            payload = pipeline._request_json("https://eutils.ncbi.nlm.nih.gov/test", retries=1)

        self.assertEqual(payload["esearchresult"]["querytranslation"], "alpha\tbeta")

    def test_request_json_reports_non_json_response_body(self):
        response = io.BytesIO(b"<html>temporary upstream error</html>")
        with patch.object(pipeline.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "Invalid JSON response from www.ebi.ac.uk") as error:
                pipeline._request_json("https://www.ebi.ac.uk/test", retries=1)

        self.assertIn("temporary upstream error", str(error.exception))

    def test_fetch_targets_spatial_biology_and_gene_regulation(self):
        result = {
            "resultList": {
                "result": [{
                    "id": "1",
                    "firstPublicationDate": "2022-03-01",
                    "title": "Spatial transcriptomics",
                    "abstractText": "A study of gene regulation.",
                }]
            }
        }
        with patch.object(pipeline, "_request_json", return_value=result):
            documents, report = pipeline.fetch_europe_pmc(2022, 2023, 1, 17)

        self.assertEqual(len(documents), 1)
        self.assertIn('"spatial transcriptomics"', report["query"])
        self.assertIn('"gene regulation"', report["query"])
        self.assertIn(' OR TITLE_ABS:(', report["query"])
        self.assertTrue(report["query"].startswith("(TITLE_ABS:("))
        self.assertIn("2022-01-01 TO 2023-12-31", report["query"])

    def test_fetch_uses_cursor_pagination_for_multiple_pages(self):
        first_page = {
            "resultList": {"result": [{
                "id": "1",
                "source": "MED",
                "firstPublicationDate": "2022-03-01",
                "title": "First spatial biology paper",
                "abstractText": "A study of gene regulation.",
            }]},
            "nextCursorMark": "second-page",
        }
        second_page = {
            "resultList": {"result": [{
                "id": "2",
                "source": "MED",
                "firstPublicationDate": "2023-04-01",
                "title": "Second spatial biology paper",
                "abstractText": "Another study of gene regulation.",
            }]},
            "nextCursorMark": "third-page",
        }
        with patch.object(
            pipeline, "_request_json", side_effect=[first_page, second_page]
        ) as request_json:
            documents, report = pipeline.fetch_europe_pmc(2022, 2023, 2, 17)

        self.assertEqual(len(documents), 2)
        self.assertEqual(report["selected_records"], 2)
        self.assertEqual(request_json.call_count, 2)
        first_query = urllib.parse.parse_qs(
            urllib.parse.urlsplit(request_json.call_args_list[0].args[0]).query
        )
        second_query = urllib.parse.parse_qs(
            urllib.parse.urlsplit(request_json.call_args_list[1].args[0]).query
        )
        self.assertEqual(first_query["cursorMark"], ["*"])
        self.assertEqual(second_query["cursorMark"], ["second-page"])
        self.assertNotIn("page", first_query)

    def test_targeted_relation_search_pages_includes_preprints_and_deduplicates_versions(self):
        preprint = {
            "id": "PPR1",
            "source": "PPR",
            "doi": "10.1101/preprint",
            "firstPublicationDate": "2024-02-01",
            "title": "Alpha regulates beta in cells",
            "abstractText": "Alpha regulates beta.",
            "pubType": "preprint",
        }
        journal_article = {
            "id": "123",
            "pmid": "123",
            "source": "MED",
            "doi": "10.1000/journal",
            "firstPublicationDate": "2024-06-01",
            "title": "Alpha regulates beta in cells",
            "abstractText": "Alpha regulates beta.",
            "pubType": "research-article; journal article",
        }
        responses = [
            {"resultList": {"result": [preprint]}, "nextCursorMark": "next"},
            {"resultList": {"result": [journal_article]}, "nextCursorMark": "next"},
        ]
        with patch.object(pipeline, "_request_json", side_effect=responses) as request_json:
            documents, report = pipeline.fetch_relation_documents(
                [{"source": "alpha", "target": "beta"}], 2024
            )

        self.assertEqual(request_json.call_count, 2)
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["pmid"], "123")
        self.assertEqual(documents[0]["source"], "MED")
        self.assertEqual(
            documents[0]["peer_review_status"],
            "journal_article_peer_review_unverified",
        )
        self.assertEqual(set(documents[0]["record_ids"]), {"PPR1", "123"})
        self.assertEqual(report["candidate_groups_searched"], 1)
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertTrue(report["preprints_included"])
        self.assertTrue(report["review_articles_excluded"])
        query = pipeline.urllib.parse.unquote_plus(request_json.call_args_list[0].args[0])
        self.assertIn('TITLE_ABS:("alpha")', query)
        self.assertIn('TITLE_ABS:("beta")', query)
        self.assertIn("NOT PUB_TYPE:review", query)
        self.assertIn("2024-01-01 TO 2024-12-31", query)

    def test_auto_targeted_search_falls_back_to_pubmed_after_europe_pmc_failure(self):
        candidate = {"source": "alpha", "target": "beta"}
        pubmed_document = pipeline.normalize_document({
            "id": "42",
            "pmid": "42",
            "source": "MED",
            "doi": "10.1000/example",
            "firstPublicationDate": "2024-02-01",
            "title": "Alpha regulates beta",
            "abstractText": "Alpha regulates beta.",
            "pubType": "Journal Article",
        })
        with patch.object(
            pipeline,
            "_request_json",
            side_effect=RuntimeError("Europe PMC unavailable"),
        ), patch.object(
            pipeline,
            "_fetch_pubmed_relation_documents",
            return_value=([pubmed_document], "pubmed query"),
        ):
            documents, report = pipeline.fetch_relation_documents([candidate], 2024)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["retrieval_provider"], "pubmed")
        self.assertEqual(report["requested_source"], "auto")
        self.assertEqual(report["sources_used"], ["pubmed"])
        self.assertEqual(report["fallback_groups"], 1)
        self.assertIn("Europe PMC unavailable", report["groups"][0]["fallback_reason"])
        self.assertEqual(report["groups"][0]["query"], "pubmed query")

    def test_pubmed_search_fetches_and_normalizes_abstract_and_publication_metadata(self):
        esearch = {"esearchresult": {"count": "1", "idlist": ["42"]}}
        efetch = """<PubmedArticleSet><PubmedArticle>
          <MedlineCitation><PMID>42</PMID><Article>
            <Journal><Title>Example Journal</Title><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>
            <ArticleTitle>Alpha regulates beta</ArticleTitle>
            <ArticleDate DateType="Electronic"><Year>2024</Year><Month>2</Month><Day>3</Day></ArticleDate>
            <Abstract><AbstractText Label="BACKGROUND">Alpha regulates beta.</AbstractText></Abstract>
            <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
          </Article></MedlineCitation><PubmedData><ArticleIdList>
            <ArticleId IdType="doi">10.1000/example</ArticleId>
            <ArticleId IdType="pmc">PMC123</ArticleId>
          </ArticleIdList></PubmedData>
        </PubmedArticle></PubmedArticleSet>"""
        with patch.object(pipeline, "_request_json", return_value=esearch) as search, patch.object(
            pipeline, "_request_text", return_value=efetch
        ) as fetch, patch.object(pipeline.time, "sleep"):
            documents, query = pipeline._fetch_pubmed_relation_documents("alpha", "beta", 2024)

        self.assertEqual(len(documents), 1)
        document = documents[0]
        self.assertEqual(document["year"], "2024")
        self.assertEqual(document["published_date"], "2024-02-03")
        self.assertEqual(document["pmid"], "42")
        self.assertEqual(document["doi"], "10.1000/example")
        self.assertEqual(document["pmcid"], "PMC123")
        self.assertIn("BACKGROUND: Alpha regulates beta.", document["abstract"])
        self.assertEqual(document["source"], "MED")
        self.assertEqual(search.call_count, 1)
        self.assertEqual(fetch.call_count, 1)
        self.assertIn('"alpha"[Title/Abstract]', query)
        self.assertIn('"beta"[Title/Abstract]', query)
        self.assertIn('"2024/01/01"[Date - Publication]', query)
        fetch_url = pipeline.urllib.parse.unquote_plus(fetch.call_args.args[0])
        self.assertIn("db=pubmed", fetch_url)
        self.assertIn("id=42", fetch_url)

    def test_normalize_preprint_does_not_expose_preprint_id_as_pmid(self):
        document = pipeline.normalize_document({
            "id": "PPR123",
            "source": "PPR",
            "firstPublicationDate": "2024-01-02",
            "pubType": "preprint",
        })

        self.assertEqual(document["pmid"], "")
        self.assertEqual(document["record_id"], "PPR123")
        self.assertEqual(document["peer_review_status"], "preprint_not_peer_reviewed")

    def test_network_check_probes_europe_pmc_and_both_ncbi_eutils(self):
        esearch = {"esearchresult": {"count": "1", "idlist": ["38892313"]}}
        efetch = """<PubmedArticleSet><PubmedArticle>
          <MedlineCitation><PMID>38892313</PMID><Article>
            <ArticleTitle>Example record</ArticleTitle>
            <Abstract><AbstractText>Example abstract.</AbstractText></Abstract>
          </Article></MedlineCitation>
        </PubmedArticle></PubmedArticleSet>"""
        with patch.object(
            pipeline,
            "_request_json",
            side_effect=[{"hitCount": 1}, esearch],
        ) as json_request, patch.object(pipeline, "_request_text", return_value=efetch) as text_request:
            results = pipeline.check_network_access()

        self.assertEqual(results["europe_pmc"]["status"], "ok")
        self.assertEqual(results["pubmed_eutils"]["status"], "ok")
        self.assertEqual(results["pubmed_eutils"]["fetched_abstract_records"], 1)
        self.assertEqual(json_request.call_count, 2)
        self.assertEqual(text_request.call_count, 1)

    def test_targeted_validate_marks_capped_candidates_as_not_searched(self):
        candidate = {"method": "ours", "rank": 1, "source": "alpha", "target": "beta"}
        unsearched_candidate = {"method": "ours", "rank": 2, "source": "gamma", "target": "delta"}
        result = {
            "id": "1",
            "pmid": "1",
            "source": "MED",
            "doi": "10.1000/example",
            "firstPublicationDate": "2024-02-01",
            "title": "Alpha and beta",
            "abstractText": "Alpha regulates beta.",
            "pubType": "research-article; journal article",
        }
        with tempfile.TemporaryDirectory() as directory:
            candidate_path = Path(directory) / "candidates.json"
            output_path = Path(directory) / "evidence.json"
            candidates = [candidate, unsearched_candidate]
            pipeline.write_json(candidate_path, {"candidates": candidates})
            pipeline.write_json(candidate_path.with_suffix(".manifest.json"), {
                "candidate_sha256": pipeline.candidate_sha256(candidates),
            })
            with patch.object(pipeline, "_request_json", return_value={
                "resultList": {"result": [result]},
                "nextCursorMark": "*",
            }), patch.object(pipeline, "LocalQwen", return_value=FakeEvidenceQwen()):
                pipeline.main([
                    "validate",
                    "--candidates", str(candidate_path),
                    "--targeted-search",
                    "--targeted-max-candidates", "1",
                    "--output", str(output_path),
                ])

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["evidence"][0]["evidence"][0]["final_label"], "uncertain")
        self.assertEqual(payload["review_model"], "local_qwen")
        self.assertEqual(payload["review_prompt_version"], pipeline.EVIDENCE_REVIEW_PROMPT_VERSION)
        self.assertEqual(payload["evidence"][1]["evidence"][0]["final_label"], "not_searched")
        self.assertEqual(payload["targeted_search"]["candidate_groups_not_searched"], 1)

    def test_validate_accepts_staged_targeted_search_and_marks_unqueried_pairs(self):
        first = {"method": "ours", "rank": 1, "source": "alpha", "target": "beta"}
        second = {"method": "ours", "rank": 2, "source": "gamma", "target": "delta"}
        documents = [{
            "year": "2024",
            "pmid": "42",
            "doi": "10.1000/example",
            "published_date": "2024-02-01",
            "title": "Alpha regulates beta",
            "abstract": "Alpha regulates beta.",
            "source": "MED",
            "retrieval_provider": "pubmed",
            "publication_type": "Journal Article",
            "peer_review_status": "journal_article_peer_review_unverified",
        }]
        with tempfile.TemporaryDirectory() as directory:
            candidate_path = Path(directory) / "candidates.json"
            validation_path = Path(directory) / "targeted.jsonl"
            report_path = Path(directory) / "targeted.search_report.json"
            output_path = Path(directory) / "evidence.json"
            candidates = [first, second]
            pipeline.write_json(candidate_path, {"candidates": candidates})
            pipeline.write_json(candidate_path.with_suffix(".manifest.json"), {
                "candidate_sha256": pipeline.candidate_sha256(candidates),
                "historical_dois": [],
            })
            pipeline.write_jsonl(validation_path, documents)
            pipeline.write_json(report_path, {
                "candidate_sha256": pipeline.candidate_sha256(candidates),
                "validation_year": 2024,
                "groups": [{"concept_group": ["alpha", "beta"], "retrieval_source": "pubmed"}],
            })
            with patch.object(pipeline, "LocalQwen", return_value=FakeEvidenceQwen()):
                pipeline.main([
                    "validate",
                    "--candidates", str(candidate_path),
                    "--validation", str(validation_path),
                    "--search-report", str(report_path),
                    "--output", str(output_path),
                ])
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["evidence"][1]["evidence"][0]["final_label"], "not_searched")
        self.assertEqual(payload["targeted_search"]["candidate_groups_not_searched"], 1)
        self.assertEqual(
            payload["evidence"][0]["evidence"][0]["retrieval_provider"],
            "pubmed",
        )

    def test_fetch_targeted_writes_abstracts_and_candidate_bound_report(self):
        candidate = {"method": "ours", "rank": 1, "source": "alpha", "target": "beta"}
        documents = [{"pmid": "42", "year": "2024", "retrieval_provider": "pubmed"}]
        with tempfile.TemporaryDirectory() as directory:
            candidate_path = Path(directory) / "candidates.json"
            output_path = Path(directory) / "targeted.jsonl"
            candidates = [candidate]
            digest = pipeline.candidate_sha256(candidates)
            pipeline.write_json(candidate_path, {"candidates": candidates})
            pipeline.write_json(candidate_path.with_suffix(".manifest.json"), {
                "candidate_sha256": digest,
            })
            with patch.object(
                pipeline,
                "fetch_relation_documents",
                return_value=(documents, {"groups": [{"concept_group": ["alpha", "beta"]}]}),
            ):
                pipeline.main([
                    "fetch-targeted",
                    "--candidates", str(candidate_path),
                    "--output", str(output_path),
                ])

            report_path = output_path.with_suffix(".search_report.json")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            fetched = pipeline.read_jsonl(output_path)

        self.assertEqual(report["candidate_sha256"], digest)
        self.assertEqual(report["validation_year"], 2024)
        self.assertEqual(fetched, documents)

    def test_cli_defaults_match_requested_corpus_and_candidate_sizes(self):
        parser = pipeline._build_parser()
        network_check = parser.parse_args(["network-check"])
        fetch_targeted = parser.parse_args([
            "fetch-targeted",
            "--candidates", "candidates.json",
            "--output", "targeted.jsonl",
        ])
        fetch = parser.parse_args(["fetch", "--output", "papers.jsonl"])
        generate = parser.parse_args(["generate", "--input", "history.jsonl", "--graph", "graph.json", "--output", "candidates.json"])
        learn = parser.parse_args(["learn", "--candidates", "candidates.json", "--evidence", "evidence.json", "--graph", "graph.json", "--output", "learned.json"])
        validate = parser.parse_args(["validate", "--candidates", "candidates.json", "--output", "evidence.json", "--targeted-search"])

        self.assertEqual(network_check.command, "network-check")
        self.assertEqual(fetch_targeted.targeted_source, "auto")
        self.assertEqual((fetch.year_start, fetch.year_end, fetch.limit), (2022, 2023, 1000))
        self.assertEqual(generate.count, 500)
        self.assertEqual(learn.count, 500)
        self.assertTrue(validate.targeted_search)
        self.assertEqual(validate.targeted_source, "auto")
        self.assertEqual(validate.targeted_max_candidates, 0)
        self.assertEqual(validate.qwen_model, pipeline.DEFAULT_QWEN_MODEL)

    def test_local_qwen_judgment_uses_sentence_only_prompt(self):
        qwen = FakeEvidenceQwen('{"label":"support","reason":"explicitly says regulates"}')

        judgment = pipeline._qwen_judgment(
            "Alpha regulates beta.", "alpha", "beta", qwen
        )

        self.assertEqual(judgment, ("support", "explicitly says regulates"))
        self.assertIn("Use only the supplied sentence", qwen.prompts[0])
        self.assertIn("do not use prior knowledge", qwen.prompts[0])
        self.assertIn("Untrusted evidence sentence", qwen.prompts[0])

    def test_network_error_explains_offline_workaround(self):
        with patch.object(
            pipeline.urllib.request,
            "urlopen",
            side_effect=pipeline.urllib.error.URLError("Network is unreachable"),
        ), patch.object(pipeline.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "retrieve targeted abstracts") as error:
                pipeline._request_json("https://www.ebi.ac.uk/test", retries=1)

        self.assertIn("network probe on the compute node", str(error.exception))

    def test_aliases_are_canonicalized(self):
        self.assertEqual(pipeline.canonicalize_concept(" PD-L1 "), "programmed death-ligand 1")
        self.assertTrue(pipeline._concept_mentioned("programmed death-ligand 1", "PD-L1 blockade"))

    def test_graph_uses_document_frequency_and_positive_pmi(self):
        documents = [
            {"concepts": ["alpha", "beta"]},
            {"concepts": ["alpha"]},
            {"concepts": ["beta"]},
        ]
        graph = pipeline.build_graph(documents, FakeEncoder())
        self.assertEqual(graph["document_frequency"], {"alpha": 2, "beta": 2})
        self.assertEqual(graph["pair_frequency"]["alpha\tbeta"], 1)
        self.assertAlmostEqual(pipeline._pair_stats(graph, "alpha", "beta")[1], math.log(0.75))
        self.assertEqual(graph["edges"], [])

    def test_graph_preserves_extracted_claim_evidence(self):
        documents = [{
            "year": "2023",
            "pmid": "7",
            "doi": "10.1000/example",
            "published_date": "2023-02-01",
            "concepts": ["alpha", "beta"],
            "claims": [{
                "subject": "alpha",
                "relation": "regulates",
                "object": "beta",
                "evidence_sentence": "Alpha regulates beta.",
            }],
        }]

        graph = pipeline.build_graph(documents, FakeEncoder())

        self.assertEqual(graph["claim_evidence"][0]["sentence"], "Alpha regulates beta.")
        self.assertEqual(graph["claim_evidence"][0]["doi"], "10.1000/example")

    def test_learn_relation_patterns_uses_historical_and_validation_evidence(self):
        historical_claim = {
            "subject": "alpha",
            "relation": "regulates",
            "object": "gamma",
            "sentence": "Alpha regulates gamma in a spatially dependent manner.",
            "pmid": "31",
        }
        group = {
            "candidate": {
                "method": "ours",
                "source": "alpha",
                "target": "gamma",
                "historical_claim_evidence": [historical_claim],
            },
            "concept_group": ["alpha", "gamma"],
            "supporting_article_count": 3,
            "supporting_evidence": [
                {"pmid": str(index), "sentence": f"Alpha regulates gamma in validation paper {index}."}
                for index in range(1, 4)
            ],
        }
        graph = {
            "documents": 20,
            "document_frequency": {"alpha": 2, "beta": 2, "gamma": 2},
            "pair_frequency": {},
            "embeddings": {"alpha": [1.0, 0.0], "beta": [0.0, 1.0], "gamma": [-1.0, 0.0]},
            "edges": [],
        }
        qwen = FakeRelationQwen()

        result = pipeline.learn_relation_candidates([group], graph, qwen, FakeRelationEncoder(), 10)

        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["method"], "ours_relation_learned")
        self.assertEqual(result["candidates"][0]["source"], "alpha")
        self.assertEqual(result["candidates"][0]["target"], "beta")
        self.assertIn("Historical claim:", qwen.prompts[0])
        self.assertIn("Independent validation", qwen.prompts[0])
        self.assertIn("Learned relation pattern:", qwen.prompts[1])

    def test_generation_produces_three_comparable_method_lists(self):
        graph = {
            "documents": 20,
            "document_frequency": {"alpha": 2, "beta": 2, "gamma": 2},
            "pair_frequency": {"alpha\tgamma": 1, "beta\tgamma": 1},
            "embeddings": {"alpha": [1.0, 0.0], "beta": [0.0, 1.0], "gamma": [-1.0, 0.0]},
            "edges": [
                {"source": "alpha", "target": "gamma", "pmi": 2.0},
                {"source": "beta", "target": "gamma", "pmi": 2.0},
            ],
        }
        candidates = pipeline.generate_candidates(graph, 10, FakeQwen())
        methods = {candidate["method"] for candidate in candidates}
        self.assertEqual(methods, {"direct_llm", "graph_path", "ours"})
        self.assertTrue(all(item["historical_cooccurrence"] == 0 for item in candidates))
        self.assertTrue(all(item["semantic_distance"] >= 0.25 for item in candidates))

    def test_generation_emits_marked_fallback_when_all_filters_reject_pairs(self):
        graph = {
            "documents": 1,
            "document_frequency": {"alpha": 1, "beta": 1},
            "pair_frequency": {"alpha\tbeta": 1},
            "embeddings": {"alpha": [1.0, 0.0], "beta": [1.0, 0.0]},
            "edges": [],
        }

        candidates = pipeline.generate_candidates(graph, 10, FakeQwen())

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["method"], "ours")
        self.assertTrue(candidates[0]["fallback"])
        self.assertEqual(candidates[0]["generation_reason"], "least_associated_fallback_thresholds_relaxed")

    def test_validation_does_not_accept_wrong_year(self):
        with self.assertRaisesRegex(ValueError, "only 2024"):
            pipeline.validate_candidates([], [{"year": "2025"}], FakeEvidenceQwen())

    def test_local_qwen_is_the_validation_reviewer_without_api_key(self):
        candidate = {"method": "ours", "source": "alpha", "target": "beta"}
        documents = [{
            "year": "2024",
            "pmid": "1",
            "doi": "10.1/example",
            "published_date": "2024-02-01",
            "title": "",
            "abstract": "Alpha interacts with beta in tumor cells.",
        }]
        qwen = FakeEvidenceQwen('{"label":"support","reason":"explicit relation"}')
        rows = pipeline.validate_candidates([candidate], documents, qwen)

        self.assertEqual(rows[0]["evidence"][0]["final_label"], "supported")
        self.assertEqual(rows[0]["evidence"][0]["review_model_label"], "support")
        self.assertEqual(len(qwen.prompts), 1)
        self.assertEqual(
            pipeline.extract_ours_supported_groups(rows, "hash", 2024),
            [],
        )

    def test_local_qwen_failure_is_not_counted_as_support(self):
        candidate = {"source": "alpha", "target": "beta"}
        documents = [{
            "year": "2024",
            "pmid": "1",
            "doi": "10.1/example",
            "published_date": "2024-02-01",
            "title": "",
            "abstract": "Alpha interacts with beta in tumor cells.",
        }]
        qwen = FakeEvidenceQwen(error=RuntimeError("model failed"))
        rows = pipeline.validate_candidates([candidate], documents, qwen)
        self.assertEqual(rows[0]["evidence"][0]["final_label"], "review_error")
        self.assertEqual(rows[0]["evidence"][0]["review_model_label"], "review_error")

    def test_extracts_ours_groups_with_three_distinct_model_supported_articles(self):
        two_support_candidate = {"method": "ours", "source": "alpha", "target": "beta"}
        three_support_candidate = {"method": "ours", "source": "gamma", "target": "delta"}
        evidence_rows = [
            {
                "candidate": two_support_candidate,
                "evidence": [
                    {"final_label": "co_mention_only"},
                    {"final_label": "supported", "sentence": "Alpha regulates beta.", "pmid": "11"},
                    {"final_label": "supported", "sentence": "Alpha activates beta.", "pmid": "12"},
                    {"final_label": "supported", "sentence": "Alpha increases beta.", "pmid": "11"},
                ],
            },
            {
                "candidate": three_support_candidate,
                "evidence": [
                    {"final_label": "supported", "sentence": "Gamma regulates delta.", "pmid": "21"},
                    {"final_label": "supported", "sentence": "Gamma activates delta.", "pmid": "22"},
                    {"final_label": "supported", "sentence": "Gamma represses delta.", "pmid": "23"},
                ],
            },
            {
                "candidate": {"method": "ours", "source": "eta", "target": "theta"},
                "evidence": [{
                    "final_label": "supported",
                    "review_model_label": "support",
                    "sentence": "Eta regulates theta.",
                    "pmid": "31",
                }] * 3,
            },
            {
                "candidate": {"method": "graph_path", "source": "epsilon", "target": "zeta"},
                "evidence": [{"final_label": "supported"}] * 3,
            },
        ]

        groups = pipeline.extract_ours_supported_groups(evidence_rows, "hash", 2024)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["concept_group"], ["gamma", "delta"])
        self.assertEqual(groups[0]["candidate_sha256"], "hash")
        self.assertEqual(groups[0]["validation_year"], 2024)
        self.assertEqual(groups[0]["supporting_article_count"], 3)
        self.assertEqual(len(groups[0]["supporting_evidence"]), 3)

    def test_extract_supported_learned_relation_groups(self):
        evidence_rows = [{
            "candidate": {
                "method": "ours_relation_learned",
                "source": "spatial omics",
                "target": "tissue architecture",
            },
            "evidence": [
                {
                    "review_model_label": "support",
                    "final_label": "supported",
                    "doi": f"10.1000/example-{index}",
                }
                for index in range(3)
            ],
        }]

        groups = pipeline.extract_ours_supported_groups(
            evidence_rows,
            "learned-hash",
            2025,
            candidate_method="ours_relation_learned",
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["method"], "ours_relation_learned")
        self.assertEqual(
            groups[0]["concept_group"],
            ["spatial omics", "tissue architecture"],
        )

    def test_manifest_detects_modified_candidate_file(self):
        candidates = [{"method": "ours", "source": "alpha", "target": "beta"}]
        with tempfile.TemporaryDirectory() as directory:
            candidate_path = Path(directory) / "candidates.json"
            pipeline.write_json(candidate_path, {"candidates": candidates})
            pipeline.write_json(candidate_path.with_suffix(".manifest.json"), {
                "candidate_sha256": pipeline.candidate_sha256(candidates),
            })
            loaded, _ = pipeline.load_and_verify_candidates(candidate_path, None)
            self.assertEqual(loaded, candidates)
            pipeline.write_json(candidate_path, {"candidates": [{**candidates[0], "target": "gamma"}]})
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                pipeline.load_and_verify_candidates(candidate_path, None)

    def test_learning_accepts_verified_ours_subset_evidence(self):
        candidates = [
            {"method": "ours", "source": "alpha", "target": "beta"},
            {"method": "direct_llm", "source": "gamma", "target": "delta"},
            {"method": "ours", "source": "epsilon", "target": "zeta"},
        ]
        ours_candidates = [
            candidate for candidate in candidates if candidate["method"] == "ours"
        ]
        rows = [{"candidate": candidate, "evidence": []} for candidate in ours_candidates]
        manifest = {"candidate_sha256": pipeline.candidate_sha256(candidates)}
        evidence_payload = {
            "candidate_sha256": pipeline.candidate_sha256(ours_candidates),
            "source_candidate_sha256": manifest["candidate_sha256"],
            "evidence": rows,
        }

        self.assertEqual(
            pipeline.validate_learning_evidence(evidence_payload, candidates, manifest),
            rows,
        )

        evidence_payload["evidence"] = rows[:1]
        with self.assertRaisesRegex(ValueError, "do not exactly match"):
            pipeline.validate_learning_evidence(evidence_payload, candidates, manifest)

    def test_learning_still_accepts_full_candidate_evidence(self):
        candidates = [
            {"method": "ours", "source": "alpha", "target": "beta"},
            {"method": "graph_path", "source": "gamma", "target": "delta"},
        ]
        rows = [{"candidate": candidate, "evidence": []} for candidate in candidates]
        manifest = {"candidate_sha256": pipeline.candidate_sha256(candidates)}
        evidence_payload = {
            "candidate_sha256": manifest["candidate_sha256"],
            "evidence": rows,
        }

        self.assertEqual(
            pipeline.validate_learning_evidence(evidence_payload, candidates, manifest),
            rows,
        )

    def test_metrics_report_common_k_and_separate_co_mentions(self):
        candidates = [
            {"method": "direct_llm", "rank": 1, "source": "a", "target": "b", "historical_cooccurrence": 0, "historical_pmi": None, "semantic_distance": 0.8},
            {"method": "graph_path", "rank": 1, "source": "c", "target": "d", "historical_cooccurrence": 0, "historical_pmi": None, "semantic_distance": 0.8},
            {"method": "graph_path", "rank": 2, "source": "e", "target": "f", "historical_cooccurrence": 0, "historical_pmi": None, "semantic_distance": 0.8},
            {"method": "ours", "rank": 1, "source": "g", "target": "h", "historical_cooccurrence": 0, "historical_pmi": None, "semantic_distance": 0.8},
            {"method": "ours", "rank": 2, "source": "i", "target": "j", "historical_cooccurrence": 0, "historical_pmi": None, "semantic_distance": 0.8},
        ]
        evidence = [{
            "candidate": candidates[0],
            "evidence": [{"final_label": "co_mention_only"}],
        }]
        report = pipeline.compute_metrics(candidates, evidence, 50, 2)
        self.assertEqual(report["metrics"]["direct_llm"]["common_k_used"], 1)
        self.assertEqual(report["metrics"]["direct_llm"]["evidence_support_rate_strict"], 0.0)
        self.assertEqual(report["metrics"]["direct_llm"]["co_mention_rate"], 1.0)

    def test_metrics_count_local_model_support_without_human_review(self):
        candidate = {
            "method": "ours",
            "rank": 1,
            "source": "alpha",
            "target": "beta",
            "historical_cooccurrence": 0,
            "historical_pmi": None,
            "semantic_distance": 0.8,
        }
        evidence = [{
            "candidate": candidate,
            "evidence": [{
                "review_model_label": "support",
                "final_label": "supported",
            }],
        }]

        report = pipeline.compute_metrics([candidate], evidence, 1, 1)

        self.assertEqual(report["metrics"]["ours"]["strict_supported_relations"], 1)
        self.assertIn("Local Qwen", report["support_definition"])

    def test_synthetic_end_to_end_generation_validation_and_metrics(self):
        history = [
            {"year": "2022", "concepts": ["alpha", "gamma"]},
            {"year": "2022", "concepts": ["beta", "gamma"]},
            {"year": "2023", "concepts": ["alpha"]},
            {"year": "2023", "concepts": ["beta"]},
        ]
        graph = pipeline.build_graph(history, FakeEncoder())
        candidates = pipeline.generate_candidates(graph, 10, FakeQwen())
        validation = [{
            "year": "2024",
            "pmid": "42",
            "doi": "10.1000/demo",
            "published_date": "2024-06-01",
            "title": "",
            "abstract": "Alpha and beta are discussed in this study.",
        }]
        evidence = pipeline.validate_candidates(candidates, validation, FakeEvidenceQwen())
        report = pipeline.compute_metrics(candidates, evidence, 1, 1)
        self.assertEqual(set(report["metrics"]), {"direct_llm", "graph_path", "ours"})
        self.assertEqual(report["validation_year"], 2024)
        self.assertTrue(all(metric["evidence_support_rate_strict"] == 0.0 for metric in report["metrics"].values()))


if __name__ == "__main__":
    unittest.main()