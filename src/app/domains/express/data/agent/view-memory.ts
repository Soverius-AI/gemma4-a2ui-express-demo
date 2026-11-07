import type { A2UIOperation } from "@copilotkit/shared";
import { bySurface, createsSurface } from "./surfaces";

const SOURCE_LIMIT = 6000;
const REQUEST_LIMIT = 800;

/** What each committed view was built from: its Express, for follow-up
 * edits, and the user requests behind it, so an action answer knows facts
 * such as a total that no submitted value holds. */
export class ViewMemory {
  readonly #sources = new Map<string, string>();
  readonly #requests = new Map<string, string>();

  source(surfaceId: string) {
    return this.#sources.get(surfaceId);
  }

  request(surfaceId: string) {
    return this.#requests.get(surfaceId);
  }

  /** Records a committed answer. `userRequest` is empty for an action
   * answer, which does not change the request a view was built for. */
  remember(operations: A2UIOperation[], dsl: string, userRequest: string) {
    // Each view's own Express, bounded, without the transport tags.
    const block = dsl.replace(/<\/?a2ui>/g, "").trim();
    for (const [id, surfaceOperations] of bySurface(operations)) {
      const created = createsSurface(surfaceOperations);
      this.#sources.set(
        id,
        created
          ? block.slice(-SOURCE_LIMIT)
          : `${this.#sources.get(id) ?? ""}\n${block}`
              .trim()
              .slice(-SOURCE_LIMIT),
      );
      if (userRequest)
        this.#requests.set(
          id,
          created
            ? userRequest.slice(-REQUEST_LIMIT)
            : `${this.#requests.get(id) ?? ""}\nThen: ${userRequest}`
                .trim()
                .slice(-REQUEST_LIMIT),
        );
    }
  }

  clear() {
    this.#sources.clear();
    this.#requests.clear();
  }
}
